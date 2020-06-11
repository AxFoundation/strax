from concurrent import futures
from functools import partial
import logging
import typing as ty
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import strax
export, __all__ = strax.exporter()

try:
    import npshmex
    SHMExecutor = npshmex.ProcessPoolExecutor
    npshmex.register_array_wrapper(strax.Chunk, 'data')
except ImportError:
    # This is allowed to fail, it only crashes if allow_shm = True
    SHMExecutor = None


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, callable]
    savers:  ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]


class MailboxDict(dict):
    def __init__(self, *args, lazy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy = lazy

    def __missing__(self, key):
        res = self[key] = strax.Mailbox(name=key + '_mailbox',
                                        lazy=self.lazy)
        return res


@export
class ThreadedMailboxProcessor:
    mailboxes: ty.Dict[str, strax.Mailbox]

    def __init__(self,
                 components: ProcessorComponents,
                 allow_rechunk=True, allow_shm=False,
                 allow_multiprocess=False,
                 allow_lazy=True,
                 max_workers=None,
                 max_messages=4,
                 timeout=60):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components

        self.log.debug("Processor components are: " + str(components))

        if allow_multiprocess and os.name == 'nt':
            print("You're on Windows! "
                  "Multiprocessing disabled, here be dragons.")
            allow_multiprocess = False

        if max_workers in [None, 1]:
            # Disable the executors: work in one process.
            # Each plugin works completely in its own thread.
            self.process_executor = self.thread_executor = None
            lazy = allow_lazy
        else:
            lazy = False
            # Use executors for parallelization of computations.
            self.thread_executor = futures.ThreadPoolExecutor(
                max_workers=max_workers)

            mp_plugins = {d: p for d, p in components.plugins.items()
                          if p.parallel == 'process'}
            if allow_multiprocess and len(mp_plugins):
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

        # Figure which ouputs
        #  - we should exclude from the flow control in lazy mode,
        #    because they are produced but not required.
        #  - we should discard (produced but neither required not saved)
        produced = set(components.loaders)
        required = set(components.targets)
        saved = set(components.savers.keys())
        for p in components.plugins.values():
            produced.update(p.provides)
            required.update(p.depends_on)
        to_flow_freely = produced - required
        to_discard = to_flow_freely - saved

        self.mailboxes = MailboxDict(lazy=lazy)

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

                self.mailboxes[mname].add_reader(
                    partial(strax.divide_outputs,
                            lazy=lazy,
                            mailboxes=self.mailboxes,
                            flow_freely=to_flow_freely,
                            outputs=p.provides))

            else:
                self.mailboxes[d].add_sender(
                    p.iter(
                        iters={dep: self.mailboxes[dep].subscribe()
                               for dep in p.depends_on},
                        executor=executor),
                    name=f'build:{d}')

        dtypes_built = {d: p
                        for p in components.plugins.values()
                        for d in p.provides}
        for d, savers in components.savers.items():
            for s_i, saver in enumerate(savers):
                if d in dtypes_built:
                    can_drive = not lazy
                    rechunk = (dtypes_built[d].can_rechunk(d)
                               and allow_rechunk)
                else:
                    # This is storage conversion mode
                    # TODO: Don't know how to get this info, for now,
                    # be conservative and don't rechunk
                    can_drive = True
                    rechunk = False

                self.mailboxes[d].add_reader(
                    partial(saver.save_from,
                            rechunk=rechunk,
                            # If paralellizing, use threads for saving
                            # the compressor releases the gil,
                            # and we have a lot of data transfer to do
                            executor=self.thread_executor),
                    can_drive=can_drive,
                    name=f'save_{s_i}:{d}')

        # For multi-output plugins, an output may be neither saved nor
        # required, and thus has to be discarded.
        # This should happen rarely in production (when you actually
        # care about the data, you will be saving it)
        def discarder(source):
            for _ in source:
                pass
        for d in to_discard:
            self.mailboxes[d].add_reader(
                discarder, name=f'discard_{d}')

        # Set to preferred number of maximum messages
        # TODO: may not work if plugins are inlined??
        for d, m in self.mailboxes.items():
            m.max_messages = max_messages
            m.timeout = timeout
            if d in components.plugins:
                max_m = components.plugins[d].max_messages
                if max_m is not None:
                    m.max_messages = max_m

    def iter(self):
        target = self.components.targets[0]
        final_generator = self.mailboxes[target].subscribe()

        self.log.debug("Starting threads")
        for m in self.mailboxes.values():
            m.start()

        self.log.debug(f"Yielding {target}")
        traceback, exc, reason = None, None, None

        try:
            yield from final_generator

        # GeneratorExit results from exception in caller
        # (on garbage collection, .close() is called, see PEP342)
        except (Exception, GeneratorExit) as e:
            self.log.fatal(f"Target Mailbox ({target}) killed, "
                           f"exception {type(e)}, message {e}")
            if isinstance(e, strax.MailboxKilled):
                _, exc, traceback = reason = e.args[0]
            else:
                exc = e
                reason = (e.__class__, e, sys.exc_info()[2])
                traceback = reason[2]

            # We will reraise it in just a moment...

        if exc is not None:
            if isinstance(exc, GeneratorExit):
                print("Main generator exited irregularly?!")
                reason[2] = (
                    "Hm, interesting. Most likely an exception was thrown "
                    "outside strax, but we did not handle it properly.")

            # Kill the mailboxes
            for m in self.mailboxes.values():
                if m != target:
                    self.log.debug(f"Killing {m}")
                    m.kill(upstream=True, reason=reason)

        self.log.debug("Closing threads")
        for m in self.mailboxes.values():
            m.cleanup()
        self.log.debug("Closing threads completed")

        self.log.debug("Closing executors")
        if self.thread_executor is not None:
            self.thread_executor.shutdown(wait=True)
        if self.process_executor not in [None, self.thread_executor]:
            self.process_executor.shutdown(wait=True)
        self.log.debug("Closing executors completed")

        if exc is not None:
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
