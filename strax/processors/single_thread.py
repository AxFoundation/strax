import typing as ty

from .base import BaseProcessor, ProcessorComponents
from .post_office import PostOffice, Spy


import strax

export, __all__ = strax.exporter()


@export
class SingleThreadProcessor(BaseProcessor):
    def __init__(
        self, components: ProcessorComponents, allow_rechunk=True, is_superrun=False, **kwargs
    ):
        super().__init__(components, allow_rechunk=allow_rechunk, is_superrun=is_superrun, **kwargs)

        self.log.debug("Processor components are: " + str(components))

        # Do not use executors: work in one thread in one process
        self.process_executor = self.thread_executor = None

        self.post_office = PostOffice()

        for d, loader in components.loaders.items():
            assert d not in components.plugins
            self.post_office.register_producer(loader(executor=self.thread_executor), topic=d)

        plugins_seen: ty.List[strax.Plugin] = []
        for d, p in components.plugins.items():
            # Multi-output plugins are listed multiple times in components.plugins;
            # ensure we only process each plugin once.
            if p in plugins_seen:
                continue
            plugins_seen.append(p)

            # Some data_types might be already saved and can be loaded;
            # remove them from the list of provides
            self.post_office.register_producer(
                p.iter(iters={dep: self.post_office.get_iter(dep, d) for dep in p.depends_on}),
                topic=strax.to_str_tuple(p.provides),
                registered=tuple(components.loaders),
            )

        dtypes_built = {d: p for p in components.plugins.values() for d in p.provides}
        for d, savers in components.savers.items():
            for saver in savers:
                if d in dtypes_built:
                    rechunk = dtypes_built[d].can_rechunk(d) and allow_rechunk
                else:
                    rechunk = is_superrun and allow_rechunk

                self.post_office.register_spy(SaverSpy(saver, rechunk=rechunk), topic=d)

    def iter(self):
        target = self.components.targets[0]
        final_generator = self.post_office.get_iter(topic=target, reader="FINAL")

        self.log.debug(f"Yielding {target}")

        try:
            yield from final_generator

        except Exception:
            # Exception in one of the producers. Close savers (they will record
            # the exception from sys.exc_info()) then reraise.
            self.log.fatal(f"Exception during processing, closing savers and reraising")
            self.post_office.kill_spies()
            raise

        except GeneratorExit:
            self.log.fatal(
                "Exception in code that called the processor: detected "
                "GeneratorExit from python shutting down. "
                "Closing savers and exiting."
            )
            # Strax savers look at sys.exc_info(). Having only "GeneratorExit"
            # there is unhelpful.. this should set it to something better:
            try:
                raise RuntimeError("Exception in caller, see log for details")
            except RuntimeError:
                self.post_office.kill_spies()

        self.log.debug("Processing finished")


class SaverSpy(Spy):
    """Spy that saves messages to a saver."""

    def __init__(self, saver, rechunk=False):
        self.saver = saver
        self.rechunker = strax.Rechunker(rechunk, self.saver.md["run_id"])
        self.chunk_number = 0

    def receive(self, chunk):
        self._save_chunk(self.rechunker.receive(chunk))

    def _save_chunk(self, chunks):
        for chunk in chunks:
            if chunk is None:
                continue
            self.saver.save(chunk, self.chunk_number)
            self.chunk_number += 1

    def close(self):
        self._save_chunk(self.rechunker.flush())
        self.saver.close()
