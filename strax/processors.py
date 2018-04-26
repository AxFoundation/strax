import logging
import typing as ty
import itertools
import threading

import numpy as np

import strax
export, __all__ = strax.exporter()



@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, ty.Iterator]
    savers:  ty.Dict[str, ty.List[strax.FileSaver]]
    sources: ty.Tuple[str]
    targets: ty.Tuple[str]



@export
class ThreadedMailboxProcessor:
    mailboxes: ty.Dict[str, strax.Mailbox]

    def __init__(self, components: ProcessorComponents, executor):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.components = components
        self.log.debug("Processor components are: " + str(components))
        plugins = components.plugins
        savers = components.savers

        # If possible, combine save and compute operations
        # so they don't have to be scheduled by executor individually.
        # This saves data transfer between cores (NUMA).
        for d, p in plugins.items():
            if not p.rechunk:
                self.log.debug(f"Putting savers for {d} in post_compute")
                for s in savers[d]:
                    p.post_compute.append(s.send)
                    p.on_close.append(s.close)     # TODO: crash close
                savers[d] = []

        # For the same reason, merge simple chains:
        # A -> B => A, with B as post_compute,
        # then put in plugins as B instead of A.
        while True:
            for b, p_b in plugins.items():
                if not p_b.rechunk and len(p_b.depends_on) == 1:
                    a = p_b.depends_on[0]
                    if a not in plugins:
                        continue
                    self.log.debug(f"Putting {b} in post_compute of {a}")
                    p_a = plugins[a]
                    p_a.post_compute.append(plugins[b].do_compute)
                    plugins[b] = p_a
                    del plugins[a]
                    break       # Changed plugins while iterating over it
            else:
                break

        self.mailboxes = {
            d: strax.Mailbox(name=d + '_mailbox')
            for d in itertools.chain.from_iterable(components)}

        for d, loader in components.loaders.items():
            assert d not in plugins
            self.mailboxes[d].add_sender(loader, name=f'load:{d}')

        for d, p in plugins.items():
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
        except strax.MailboxKilled:
            self.log.debug(f"Target Mailbox ({target}) killed")
            for m in self.mailboxes.values():
                self.log.debug(f"Killing {m}")
                m.kill(upstream=True,
                       reason="Strax terminating due to downstream exception")

        self.log.debug("Closing threads")
        for m in self.mailboxes.values():
            m.cleanup()


@export
class BackgroundThreadProcessor:
    """Same as ThreadedMailboxProcessor, but processes in a background thread
    rather than allowing you to iterate over results.

    Use as a context manager.
    """
    def __init__(self, components: ProcessorComponents, executor):
        self.log = logging.getLogger(self.__class__.__name__)
        # self.log.setLevel(logging.DEBUG)
        self.c = components
        self.p = ThreadedMailboxProcessor(components, executor)

        def _spin():
            for _ in self.p.iter():
                pass
        self.t = threading.Thread(target=_spin, name='BackgroundProcessor')

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for d in self.c.sources:
                self.p.mailboxes[d].close()
        else:
            for d in self.c.sources:
                self.p.mailboxes[d].kill()

        # Now it needs some time to finish processing stuff that was waiting
        # in buffers
        self.t.join(timeout=60)
        if self.t.is_alive():
            raise RuntimeError("Processing did not terminate?!")

    def send(self, data_type: str, data: np.ndarray, chunk_i: int):
        """Send new data numbered chunk_i of data_type"""
        self.log.debug(f"Received {chunk_i} of data type {data_type}")
        if data_type not in self.c.sources:
            raise RuntimeError(f"Can't send {data_type}, not a source.")
        self.p.mailboxes[data_type].send(msg=data, msg_number=chunk_i)
        self.log.debug(f"Sent {chunk_i} to mailbox")


@export
class SimpleChain:
    """A very basic processor that processes things on-demand
    without any background threading or rechunking.

    Suitable for high-performance multiprocessing.
    Use as a context manager.
    """
    def __init__(self, components: ProcessorComponents):
        self.log = logging.getLogger(self.__class__.__name__)
        self.c = c = components
        self.log.info(f"Simple chain components: {c}")
        if (any([len(p.depends_on) != 1 for p in c.plugins.values()])
                or len(self.c.targets) != 1
                or len(self.c.sources) != 1):
            raise RuntimeError("simple_chain only supports linear "
                               "source->target processing.")
        chain = [c.targets[0]]
        while chain[-1] != c.sources[0]:
            d = chain[-1]
            p = c.plugins[d]
            chain.append(p.depends_on[0])
        self.chain = list(reversed(chain))
        self.log.info(self.chain)

    def send(self, data: np.ndarray, chunk_i: int):
        for i, d in enumerate(self.chain):
            if i != 0:
                p = self.c.plugins[d]
                data = p.compute(data)   # Pycharm does not get this
            for s in self.c.savers.get(d, []):
                self.log.debug(f"Saving {chunk_i} of {d}")
                s.send(data=data, chunk_i=chunk_i)
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for d in self.chain:
            for s in self.c.savers.get(d, []):
                self.log.debug(f"Closing saver for {d}")
                if exc_type is None:
                    s.close()
                else:
                    s.crash_close()
