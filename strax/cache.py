from collections import namedtuple
import os
import logging

import strax
export, __all__ = strax.exporter()

CacheKey = namedtuple('CacheKey',
                      ('run_id', 'data_type', 'lineage'))
export(CacheKey)


@export
class NotCachedException(Exception):
    pass


@export
class FileCache:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _dirname(key):
        return os.path.join(key.run_id, key.data_type)

    def get(self, key):
        """Return iterator factory over cached results,
        or raise NotCachedException if we have not cached the results yet
        """
        dirname = self._dirname(key)
        if os.path.exists(dirname):
            self.log.debug(f"{key} is in cache.")
            return strax.io_chunked.read_chunks(dirname)
        self.log.debug(f"{key} is NOT in cache.")
        raise NotCachedException

    def save(self, key, source):
        dirname = self._dirname(key)
        source = strax.chunk_arrays.fixed_size_chunks(source)
        strax.io_chunked.save_to_dir(source, dirname)
