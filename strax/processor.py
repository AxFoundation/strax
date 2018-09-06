from concurrent import futures
import logging
import typing as ty

import strax
export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, ty.Iterator]
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
                 allow_rechunk=True,
                 max_workers=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components
        self.mailboxes = MailboxDict()

        self.log.debug("Processor components are: " + str(components))
        plugins = components.plugins
        savers = components.savers

        if max_workers in [None, 1]:
            # Disable the executors: work in one process.
            # Each plugin works completely in its own thread.
            process_executor = thread_executor = None
        else:
            # Use executors for parallelization of computations.
            process_executor = futures.ProcessPoolExecutor(
                max_workers=max_workers)
            thread_executor = futures.ThreadPoolExecutor(
                max_workers=max_workers)

        # Deal with parallel input processes
        # Setting up one of these modifies plugins, so we must gather
        # them all first.
        par_inputs = [p for p in plugins.values()
                      if issubclass(p.__class__, strax.ParallelSourcePlugin)]
        for p in par_inputs:
            components = p.setup_mailboxes(components,
                                           self.mailboxes,
                                           process_executor)

        self.log.debug("After optimization: " + str(components))

        for d, loader in components.loaders.items():
            assert d not in plugins
            self.mailboxes[d].add_sender(loader, name=f'load:{d}')

        for d, p in plugins.items():
            executor = None
            if p.parallel == 'process':
                executor = process_executor
            elif p.parallel:
                executor = thread_executor

            self.mailboxes[d].add_sender(p.iter(
                    iters={d: self.mailboxes[d].subscribe()
                           for d in p.depends_on},
                    executor=executor),
                name=f'build:{d}')

        for d, savers in savers.items():
            for s_i, saver in enumerate(savers):
                if d in plugins:
                    rechunk = plugins[d].rechunk_on_save
                else:
                    # This is storage conversion mode
                    # TODO: Don't know how to get this info, for now,
                    # be conservative and don't rechunk
                    rechunk = False
                if not allow_rechunk:
                    rechunk = False

                from functools import partial
                self.mailboxes[d].add_reader(
                    partial(saver.save_from, rechunk=rechunk),
                    name=f'save_{s_i}:{d}')

    def iter(self):
        target = self.components.targets[0]
        final_generator = self.mailboxes[target].subscribe()

        self.log.debug("Starting threads")
        for m in self.mailboxes.values():
            m.start()

        self.log.debug(f"Yielding {target}")
        try:
            yield from final_generator
            traceback = None
            exc = None
        except strax.MailboxKilled as e:
            self.log.debug(f"Target Mailbox ({target}) killed")
            for m in self.mailboxes.values():
                if m != target:
                    self.log.debug(f"Killing {m}")
                    m.kill(upstream=True,
                           reason=e.args[0])
            _, exc, traceback = e.args[0]
        finally:
            self.log.debug("Closing threads")
            for m in self.mailboxes.values():
                m.cleanup()

        # Reraise exception. This is outside the except block
        # to avoid the 'during handling of this exception, another
        # exception occurred' stuff from confusing the traceback
        # which is printed for the user
        if traceback is not None:
            raise exc.with_traceback(traceback)
