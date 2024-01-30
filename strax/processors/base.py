import logging
import typing as ty

import strax

export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor."""

    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, ty.Callable]
    # Required for inline ParallelSource plugin.
    loader_plugins: ty.Dict[str, strax.Plugin]
    savers: ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]


@export
class BaseProcessor:
    components: ProcessorComponents

    def __init__(self, components: ProcessorComponents, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components

    def iter(self):
        raise NotImplementedError
