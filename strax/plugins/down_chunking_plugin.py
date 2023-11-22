import strax
import types
import inspect
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

    def iter(self, iters, executor=None):
        _plugin_uses_multi_threading = (
            self.parallel and executor is not None and (inspect.isgeneratorfunction(self.compute))
        )
        if _plugin_uses_multi_threading:
            raise NotImplementedError(
                f'Plugin "{self.__class__.__name__}" uses an iterator as compute method. '
                "This is not supported in multi-threading/processing."
            )
        return super().iter(iters, executor=None)

    def _iter_return(self, chunk_i, **inputs_merged):
        return self.do_compute(chunk_i=chunk_i, **inputs_merged)

    def _fix_output(self, result, start, end, _dtype=None):
        """Wrapper around _fix_output to support the return of iterators."""
        if isinstance(result, types.GeneratorType):
            return result
        return super()._fix_output(result, start, end)
