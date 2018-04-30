from concurrent import futures
import logging
import typing as ty
import itertools

import strax
export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, ty.Iterator]
    savers:  ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]


@export
class ThreadedMailboxProcessor:
    mailboxes: ty.Dict[str, strax.Mailbox]

    def __init__(self, components: ProcessorComponents, max_workers=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components
        self.log.debug("Processor components are: " + str(components))
        plugins = components.plugins
        savers = components.savers

        # Executors for parallelization
        process_executor = futures.ProcessPoolExecutor(max_workers=max_workers)
        thread_executor = futures.ThreadPoolExecutor(max_workers=max_workers)

        # If possible, combine save and compute operations
        # so they don't have to be scheduled by executor individually.
        # This saves data transfer between cores (NUMA).
        for d, p in plugins.items():
            if not p.rechunk_on_save:
                self.log.debug(f"Putting savers for {d} in post_compute")
                for s in savers.get(d, []):
                    p.post_compute.append(s.save)
                    p.on_close.append(s.close)
                savers[d] = []

        # For the same reason, merge simple chains:
        # A -> B => A, with B as post_compute,
        # then put in plugins as B instead of A.
        # TODO: check they agree on paralellization?
        # TODO: allow compute grouping while saver does rechunk
        while True:
            for b, p_b in plugins.items():
                if (p_b.parallel and not p_b.rechunk_on_save
                        and len(p_b.depends_on) == 1
                        and b not in components.targets):
                    a = p_b.depends_on[0]
                    if a not in plugins:
                        continue
                    self.log.debug(f"Putting {b} in post_compute of {a}")
                    p_a = plugins[a]
                    p_a.post_compute.append(plugins[b].do_compute)
                    p_a.on_close.extend(p_b.on_close)
                    plugins[b] = p_a
                    del plugins[a]
                    break       # Changed plugins while iterating over it
            else:
                break

        self.log.debug("After optimization: " + str(components))

        self.mailboxes = {
            d: strax.Mailbox(name=d + '_mailbox')
            for d in itertools.chain.from_iterable(components)}

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
                self.mailboxes[d].add_reader(saver.save_from,
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
        except strax.MailboxKilled as e:
            self.log.debug(f"Target Mailbox ({target}) killed")
            for m in self.mailboxes.values():
                if m != target:
                    self.log.debug(f"Killing {m}")
                    m.kill(upstream=True,
                           reason=e.args[0])

        self.log.debug("Closing threads")
        for m in self.mailboxes.values():
            m.cleanup()
