
import logging
import typing as ty

import strax
export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, callable]
    savers:  ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]
    
@export
class BaseProcessor:
    components: ProcessorComponents

    def __init__(self,
                 components: ProcessorComponents,
                 allow_rechunk=True, allow_shm=False,
                 allow_multiprocess=False,
                 allow_lazy=True,
                 max_workers=None,
                 max_messages=4,
                 timeout=60):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components

    def iter(self):
        raise NotImplementedError

