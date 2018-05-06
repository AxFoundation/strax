from ast import literal_eval
import logging
import sys
import time
import traceback
import typing

import numpy as np

import strax
export, __all__ = strax.exporter()


@export
class CacheKey(typing.NamedTuple):
    run_id: str
    data_type: str
    lineage: dict

    def __repr__(self):
        return '_'.join([self.run_id,
                         self.data_type,
                         strax.deterministic_hash(self.lineage)])


@export
class NotCached(Exception):
    pass


@export
class NotProvided(Exception):
    pass


class Store:
    """Storage backend for strax data
    """

    saver_init_doc = """\
        :param recover: Load data even if an exception occurred
            during computation (and the data is thus likely incomplete)
        :param take_only: Provide only these data types.
        :param exclude: Do NOT provide these data types.

        If take_only and exclude are both omitted, provide all data types.
        If a data type is listed in both, it will not be provided.

        Attempting to read unwanted data types throws NotCached.
        Attempting to save unwanted data types throws RuntimeError
            (you're supposed to check this with the .provides method).
    """

    def __init__(self,
                 take_only=tuple(), exclude=tuple(),
                 recover=False,
                 readonly=False):
        self._take_only = strax.to_str_tuple(take_only)
        self._exclude = strax.to_str_tuple(exclude)
        self.recover = recover
        self.readonly = readonly
        self.log = logging.getLogger(self.__class__.__name__)

    def provides(self, data_type, write=False):
        """Return whether this store can provides this datatype
        :param write: if True, return if we can also store the data (in case
        we do not have it yet)
        """
        if write and self.readonly:
            self.log.debug(f"Can't save {data_type}, we're readonly")
            return False
        if len(self._exclude) and data_type in self._exclude:
            self.log.debug(f"Not saving {data_type}, excluded")
            return False
        if len(self._take_only) and data_type not in self._take_only:
            self.log.debug(f"Not saving {data_type}, not taken")
            return False
        return True

    def has(self, key: CacheKey):
        try:
            self.find(key)
        except NotCached:
            return False
        return True

    def find(self, key: CacheKey):
        """Return something to load key from (e.g. a directory name)
        or raise NotCached
        """
        something = self._find(key)
        if 'exception' in self._read_meta(something):
            self.log.info(f"Found incomplete data for {key}")
            if self.recover:
                self.log.info("Recovering")
                return something
            self.log.info("Not recovering")
            raise NotCached
        return something

    def loader(self, key: CacheKey, executor=None):
        """Return generator over cached results,
        or raise NotCached if the data is unavailable.
        """
        return self._read(self.find(key), executor)

    def load_meta(self, key: CacheKey):
        return self._read_meta(self.find(key))

    def saver(self, key, metadata):
        if not self.provides(key.data_type):
            raise RuntimeError(f"{key.data_type} not provided by "
                               f"storage backend.")

        self.log.debug(f"Saving {key}")

        metadata.setdefault('compressor', 'blosc')
        metadata['strax_version'] = strax.__version__
        if 'dtype' in metadata:
            metadata['dtype'] = metadata['dtype'].descr.__repr__()
        # >>> Child class should finish saver creation! <<<

    def _read(self, something, executor=None):
        """Iterates over strax results in something (e.g. a directory
        or database collection)
        """
        metadata = self._read_meta(something)
        if not len(metadata['chunks']):
            self.log.warning(f"No actual data in {something}?")
        dtype = literal_eval(metadata['dtype'])
        compressor = metadata['compressor']

        for chunk_info in metadata['chunks']:
            kwargs = dict(chunk_info=chunk_info,
                          dtype=dtype,
                          compressor=compressor)

            if executor is None:
                yield self._read_chunk(something, **kwargs)
            else:
                yield executor.submit(self._read_chunk, something, **kwargs)

    def _find(self, key):
        """Return something self.read reads from, or NotCachedException"""
        raise NotImplementedError

    def _read_meta(self, something):
        raise NotImplementedError

    def _read_chunk(self, something, chunk_info, dtype, compressor):
        raise NotImplementedError


@export             # Needed for type hints elsewhere
class Saver:
    """Interface for saving a data type

    Must work even if forked.
    Do NOT add unpickleable things as attributes (such as loggers)!
    """
    closed = False
    meta_only = False
    prefer_rechunk = True

    def __init__(self, key, metadata):
        self.key = key
        self.md = metadata
        self.md['writing_started'] = time.time()
        self.md['chunks'] = []

    def save_from(self, source: typing.Iterable, rechunk=True):
        """Iterate over source and save the results under key
        along with metadata
        """
        if rechunk and self.prefer_rechunk:
            source = strax.fixed_size_chunks(source)

        try:
            for chunk_i, s in enumerate(source):
                self.save(data=s, chunk_i=chunk_i)
        except strax.MailboxKilled:
            # Write exception (with close), but exit gracefully.
            # One traceback on screen is enough
            self.close()
            pass
        finally:
            if not self.closed:
                self.close()

    def save(self, data: np.ndarray, chunk_i: int):
        if self.closed:
            raise RuntimeError(f"{self.key.data_type} saver already closed!")

        chunk_info = dict(chunk_i=chunk_i,
                          n=len(data),
                          nbytes=data.nbytes)
        if 'time' in data[0].dtype.names:
            for desc, i in (('first', 0), ('last', -1)):
                chunk_info[f'{desc}_time'] = int(data[i]['time'])
                chunk_info[f'{desc}_endtime'] = int(strax.endtime(data[i]))

        if not self.meta_only:
            chunk_info.update(self._save_chunk(data, chunk_info))
        self._save_chunk_metadata(chunk_info)

    def _save_chunk(self, data, chunk_info):
        raise NotImplementedError

    def _save_chunk_metadata(self, chunk_info):
        raise NotImplementedError

    def close(self):
        if self.closed:
            raise RuntimeError(f"{self.key.data_type} saver already closed!")
        self.closed = True
        exc_info = sys.exc_info()
        if exc_info[0] not in [None, StopIteration]:
            self.md['exception'] = traceback.format_exc()
        self.md['writing_ended'] = time.time()

        # >>> child class may want to finish writing metadata <<<
