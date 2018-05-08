from collections import defaultdict
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


def savers_to_post_compute(c, log=None, force=False):
    """Put savers in post_compute of their associated plugins
    :param log: logger
    :param force: do it even if it means disabling rechunking
    """
    # If possible, combine save and compute operations
    # so they don't have to be scheduled by executor individually.
    # This saves data transfer between cores (NUMA).
    for d, p in c.plugins.items():
        if force or not p.rechunk_on_save:
            if log:
                log.debug(f"Putting savers for {d} in post_compute")
            for s in c.savers.get(d, []):
                p.post_compute.append(s.save)
                p.on_close.append(s.close)
            del c.savers[d]

def merge_chains(c, log=None):
    # Merge simple chains:
    # A -> B => A, with B as post_compute,
    # then put in plugins as B instead of A.
    # TODO: check they agree on paralellization?
    # TODO: allow compute grouping while saver does rechunk
    while True:
        for b, p_b in c.plugins.items():
            if (p_b.parallel and not p_b.rechunk_on_save
                    and len(p_b.depends_on) == 1
                    and b not in c.targets):
                a = p_b.depends_on[0]
                if a not in c.plugins:
                    continue
                if log:
                    log.debug(f"Putting {b} in post_compute of {a}")
                p_a = c.plugins[a]
                p_a.post_compute.append(c.plugins[b].do_compute)
                p_a.on_close.extend(p_b.on_close)
                c.plugins[b] = p_a
                del c.plugins[a]
                break       # Changed plugins while iterating over it
        else:
            break
    return c



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

        savers_to_post_compute(components, log=self.log)
        merge_chains(components, log=self.log)

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


@export
class IteratorProcessor:
    """Simple processor that does not use mailboxes or threading, but
    cannot do multiprocessing"""

    def __init__(self, components: ProcessorComponents, max_workers=1):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components
        self.log.debug("Processor components are: " + str(components))

        assert len(components.targets) == 1, "Only single target supported"
        target = components.targets[0]
        assert max_workers in [1, None], "Multiprocessing not implemented"

        savers_to_post_compute(components, log=self.log, force=True)
        assert len(components.savers) == 0
        merge_chains(components, log=self.log)


        # Find out how many iterators we need for each data type
        n_iters = defaultdict(int)
        n_iters[target] += 1
        for p in components.plugins.values():
            for d in p.depends_on:
                n_iters[d] += 1

        def make_iters(d, it):
            nonlocal iters
            iters[d] = list(itertools.tee(it, n_iters[d]))

        iters = dict()
        for d, l in components.loaders.items():
            make_iters(d, l)

        plugins = components.plugins.copy()
        while len(plugins):
            for d, p in plugins.items():
                # Check we have iters for this plugin already
                try:
                    for dep in p.depends_on:
                        if dep not in iters:
                            raise KeyError
                except KeyError:
                    continue

                it = p.iter({dep: iters[dep].pop()
                             for dep in p.depends_on})
                make_iters(d, it)
                del plugins[d]
                break    # Can't modify while iterating

        self.final_iterator = iters[target].pop()
        for d, its in iters.items():
            assert len(its) == 0, f"{len(its)} unusused iterators for {d}!"

    def iter(self):
        yield from self.final_iterator

        # Close remaining savers; does not happen automatically
        # due to branching
        for p in self.components.plugins.values():
            p.close()


"""
def read_ahead(plugin, executor, n=15):

    next_chunk_i = 0
    pending = dict()
    stopped = False
    for chunk_i in itertools.count():
        # Submit new futures
        while not stopped and len(pending) < n:
            try:
                pending[next_chunk_i] = next(source_of_futures)
                next_chunk_i += 1
            except StopIteration:
                stopped = True

        # Next alread pending?
        if next_chunk_i

        # Next
"""