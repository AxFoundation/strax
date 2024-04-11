import strax
from .plugin import Plugin

export, __all__ = strax.exporter()


##
# Plugin which allows to use yield in plugins compute method.
# Allows to chunk down output before storing to disk.
# Only works if multiprocessing is omitted.
##


@export
class DownChunkingPlugin(Plugin):
    """Plugin that merges data from its dependencies."""

    parallel = False

    def __init__(self):
        super().__init__()

        if self.parallel:
            raise NotImplementedError(
                f'Plugin "{self.__class__.__name__}" is a DownChunkingPlugin which '
                "currently does not support parallel processing."
            )

        if self.multi_output:
            raise NotImplementedError(
                f'Plugin "{self.__class__.__name__}" is a DownChunkingPlugin which '
                "currently does not support multiple outputs. Please only provide "
                "a single data-type."
            )

    def _iter_compute(self, chunk_i, **inputs_merged):
        return self.do_compute(chunk_i=chunk_i, **inputs_merged)

    def _fix_output(self, result, start, end, _dtype=None):
        """Wrapper around _fix_output to support the return of iterators."""
        return result
