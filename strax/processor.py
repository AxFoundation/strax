from concurrent import futures
from functools import partial
import logging
import typing as ty
import psutil
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
try:
    from npshmex import ProcessPoolExecutor as SHMExecutor
except ImportError:
    # This is allowed to fail, it only crashes if allow_shm = True
    SHMExecutor = None
    pass

import numpy as np

import strax
export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, callable]
    savers:  ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]


class MailboxDict(dict):
    def __missing__(self, key):
        res = self[key] = strax.Mailbox(name=key + '_mailbox')
        return res


@export
class ThreadedMailboxProcessor:
    mailboxes: ty.Dict[str, strax.Mailbox]

    def __init__(self,
                 components: ProcessorComponents,
                 allow_rechunk=True, allow_shm=False,
                 allow_multiprocess=False,
                 max_workers=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components
        self.mailboxes = MailboxDict()

        self.log.debug("Processor components are: " + str(components))

        if max_workers in [None, 1]:
            # Disable the executors: work in one process.
            # Each plugin works completely in its own thread.
            self.process_executor = self.thread_executor = None
        else:
            # Use executors for parallelization of computations.
            self.thread_executor = futures.ThreadPoolExecutor(
                max_workers=max_workers)

            mp_plugins = {d: p for d, p in components.plugins.items()
                          if p.parallel == 'process'}
            if (allow_multiprocess and len(mp_plugins)):
                _proc_ex = ProcessPoolExecutor
                if allow_shm:
                    if SHMExecutor is None:
                        raise RuntimeError(
                            "You must install npshmex to enable shm"
                            " transfer of numpy arrays.")
                    _proc_ex = SHMExecutor
                self.process_executor = _proc_ex(max_workers=max_workers)

                # Combine as many plugins /savers as possible in one process
                # TODO: more intelligent start determination, multiple starts
                start_from = list(mp_plugins.keys())[
                    int(np.argmin([len(p.depends_on)
                                   for p in mp_plugins.values()]))]
                components = strax.ParallelSourcePlugin.inline_plugins(
                    components, start_from, log=self.log)
                self.components = components
                self.log.debug("Altered components for multiprocessing: "
                               + str(components))
            else:
                self.process_executor = self.thread_executor

        for d, loader in components.loaders.items():
            assert d not in components.plugins
            # If paralellizing, use threads for loading
            # the decompressor releases the gil, and we have a lot
            # of data transfer to do
            self.mailboxes[d].add_sender(
                loader(executor=self.thread_executor),
                name=f'load:{d}')

        multi_output_seen = []
        for d, p in components.plugins.items():
            if p in multi_output_seen:
                continue

            executor = None
            if p.parallel == 'process':
                executor = self.process_executor
            elif p.parallel:
                executor = self.thread_executor

            if p.multi_output:
                multi_output_seen.append(p)

                # Create temp mailbox that receives multi-output dicts
                # and sends them forth to other mailboxes
                mname = p.__class__.__name__ + '_divide_outputs'
                self.mailboxes[mname].add_sender(
                    p.iter(
                        iters={dep: self.mailboxes[dep].subscribe()
                               for dep in p.depends_on},
                        executor=executor),
                    name=f'divide_outputs:{d}')

                for d in p.provides:
                    self.mailboxes[d]   # creates mailbox d if doesn't exist
                self.mailboxes[mname].add_reader(
                    partial(strax.divide_outputs,
                            mailboxes=self.mailboxes,
                            outputs=p.provides))

            else:
                self.mailboxes[d].add_sender(
                    p.iter(
                        iters={dep: self.mailboxes[dep].subscribe()
                               for dep in p.depends_on},
                        executor=executor),
                    name=f'build:{d}')

        for d, savers in components.savers.items():
            for s_i, saver in enumerate(savers):
                if d in components.plugins:
                    rechunk = components.plugins[d].rechunk_on_save
                else:
                    # This is storage conversion mode
                    # TODO: Don't know how to get this info, for now,
                    # be conservative and don't rechunk
                    rechunk = False
                if not allow_rechunk:
                    rechunk = False

                self.mailboxes[d].add_reader(
                    partial(saver.save_from,
                            rechunk=rechunk,
                            # If paralellizing, use threads for saving
                            # the compressor releases the gil,
                            # and we have a lot of data transfer to do
                            executor=self.thread_executor),
                    name=f'save_{s_i}:{d}')

    def iter(self):
        target = self.components.targets[0]
        final_generator = self.mailboxes[target].subscribe()

        self.log.debug("Starting threads")
        for m in self.mailboxes.values():
            m.start()

        self.log.debug(f"Yielding {target}")
        traceback = None
        exc = None

        try:
            yield from final_generator

        except Exception as e:
            self.log.debug(f"Target Mailbox ({target}) killed, exception {e}")
            for m in self.mailboxes.values():
                if m != target:
                    self.log.debug(f"Killing {m}")
                    if isinstance(e, strax.MailboxKilled):
                        _, exc, traceback = reason = e.args[0]
                    else:
                        exc = e
                        reason = (e.__class__, e, sys.exc_info()[2])
                        traceback = reason[2]

                    m.kill(upstream=True, reason=reason)
            # We will reraise it in just a moment...

        finally:
            self.log.debug("Closing threads")
            for m in self.mailboxes.values():
                m.cleanup()
            self.log.debug("Closing threads completed")

            self.log.debug("Closing executors")
            if self.thread_executor is not None:
                self.thread_executor.shutdown(wait=False)

            if self.process_executor not in [None, self.thread_executor]:
                # Unfortunately there is no wait=timeout option, so we have to
                # roll our own
                pids = self.process_executor._processes.keys()
                self.process_executor.shutdown(wait=False)

                t0 = time.time()
                while time.time() < t0 + 20:
                    if all([not psutil.pid_exists(pid) for pid in pids]):
                        break
                    self.log.info("Waiting for subprocesses to end")
                    time.sleep(2)
                else:
                    self.log.warning("Subprocesses failed to terminate, "
                                     "resorting to brute force killing")
                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            # Didn't exist
                            pass
                    self.log.info("Sent SIGTERM to all subprocesses")

            self.log.debug("Closing executors completed")

        if traceback is not None:
            # Reraise exception. This is outside the except block
            # to avoid the 'during handling of this exception, another
            # exception occurred' stuff from confusing the traceback
            # which is printed for the user
            self.log.debug("Reraising exception")
            raise exc.with_traceback(traceback)

        # Check the savers for any exception that occurred during saving
        # These are thrown back to the mailbox, but if that has already closed
        # it doesn't trigger a crash...
        # TODO: add savers inlined by parallelsourceplugin
        # TODO: need to look at plugins too if we ever implement true
        # multi-target mode
        for k, saver_list in self.components.savers.items():
            for s in saver_list:
                if s.got_exception:
                    self.log.fatal(f"Caught error while saving {k}!")
                    raise s.got_exception

        self.log.debug("Processing finished")
