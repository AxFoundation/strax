import logging
import typing
import itertools
import threading

import numpy as np

import strax
export, __all__ = strax.exporter()


@export
class ThreadedMailboxProcessor:
    mailboxes: typing.Dict[str, strax.Mailbox]

    def __init__(self, components: strax.ProcessorComponents, executor):
        self.log = logging.getLogger(self.__class__.__name__)
        self.c = c = components

        self.mailboxes = {
            d: strax.Mailbox(name=d + '_mailbox')
            for d in itertools.chain.from_iterable(components)}

        for d in c.loaders:
            assert d not in c.plugins
            self.mailboxes[d].add_sender(c.loaders[d], name=f'load:{d}')

        for d in c.plugins:
            p = c.plugins[d]
            self.mailboxes[d].add_sender(p.iter(
                    iters={d: self.mailboxes[d].subscribe()
                           for d in p.depends_on},
                    executor=executor),
                name=f'build:{d}')

        for d in c.savers:
            for s_i, s in enumerate(c.savers[d]):
                self.mailboxes[d].add_reader(s.save_from,
                                             name=f'save_{s_i}:{d}')

    def iter(self):
        target = self.c.targets[0]
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
    def __init__(self, components: strax.ProcessorComponents, executor):
        self.log = logging.getLogger(self.__class__.__name__)
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

    def send(self, data_type: str, chunk_i: int, data: np.ndarray):
        """Send new data numbered chunk_i of data_type"""
        if data_type not in self.c.sources:
            raise RuntimeError(f"Can't send {data_type}, not a source.")
        self.p.mailboxes[data_type].send(data, msg_number=chunk_i)


@export
class SimpleChain:
    """A very basic processor that processes things on-demand
    without any background threading or rechunking.

    Suitable for high-performance multiprocessing.
    Use as a context manager.
    """
    def __init__(self, components: strax.ProcessorComponents):
        self.log = logging.getLogger(self.__class__.__name__)
        self.c = c = components
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
        self.chain = list(reversed(chain[:-1]))

    def send(self, chunk_i: int, data: np.ndarray):
        for d in self.chain:
            p = self.c.plugins[d]
            data = p.compute(data)   # Pycharm does not get this
            for s in self.c.savers.get(d, []):
                s.send(chunk_i=chunk_i, data=data)
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for d in self.chain:
            for s in self.c.savers.get(d, []):
                if exc_type is None:
                    s.close()
                else:
                    s.crash_close()
