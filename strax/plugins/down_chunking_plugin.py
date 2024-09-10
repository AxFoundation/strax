from typing import Generator

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

    def _iter_compute(self, chunk_i, **inputs_merged):
        return self.do_compute(chunk_i=chunk_i, **inputs_merged)

    def _fix_output(self, result, start, end, superrun, subruns, _dtype=None):
        """Wrapper around _fix_output to support the return of iterators."""
        if not isinstance(result, Generator):
            raise ValueError(
                f"Plugin {self.__class__.__name__} should return a generator in compute method."
            )

        for _result in result:
            if isinstance(_result, dict):
                values = _result.values()
            else:
                if self.multi_output:
                    raise ValueError(
                        f"{self.__class__.__name__} is multi-output and should "
                        "provide a generator of dict output."
                    )
                values = [_result]
            if not all(isinstance(v, strax.Chunk) for v in values):
                raise ValueError(
                    f"Plugin {self.__class__.__name__} should yield (dict of) "
                    "strax.Chunk in compute method."
                )
            yield self.superrun_transformation(_result, superrun, subruns)
