"""Base classes for storage backends, frontends, and savers in strax.

Please see the developer documentation for more details
on strax' storage hierarchy.
"""
import logging
import sys
import time
import traceback
import typing
import warnings
from ast import literal_eval
from concurrent.futures import wait

import numpy as np

import strax

export, __all__ = strax.exporter()


@export
class DataKey(typing.NamedTuple):
    """Request for data to a storage registry

    Instances of this class uniquely identify a single piece of strax data
    abstractly -- that is, it describes the full history of algorithms that
    have to be run to reproduce it.

    It is used for communication between the main Context class and storage
    frontends.
    """
    run_id: str
    data_type: str
    lineage: dict

    def __repr__(self):
        return '_'.join([self.run_id,
                         self.data_type,
                         strax.deterministic_hash(self.lineage)])


@export
class DataNotAvailable(Exception):
    """Raised when requested data is not available"""
    pass


@export
class AmbiguousDataRequest(Exception):
    """Raised when more than one piece of data match a users' request"""
    def __init__(self, found, message=''):
        super().__init__(message)
        self.found = found


@export
class DataExistsError(Exception):
    """Raised when attempting to write a piece of data
    that is already written"""
    def __init__(self, at, message=''):
        super().__init__(message)
        self.at = at


@export
class RunMetadataNotAvailable(Exception):
    pass


@export
class StorageFrontend:
    """Interface to something that knows data-locations and run-level metadata.
    For example, a runs database, or a data directory on the file system.
    """
    backends: list

    def __init__(self,
                 readonly=False,
                 overwrite='if_broken',
                 take_only=tuple(), exclude=tuple()):
        """
        :param readonly: If True, throws CannotWriteData whenever saving is
        attempted.
        :param overwrite: When to overwrite data that already exists.
         - 'never': Never overwrite any data.
         - 'if_broken': Only overwrites data if it is incomplete or broken.
         - 'always': Always overwrite data. Use with caution!
        :param take_only: Provide/accept only these data types.
        :param exclude: Do NOT provide/accept these data types.

        If take_only and exclude are both omitted, provide all data types.
        If a data type is listed in both, it will not be provided.
        Attempting to read/write unwanted data types throws DataTypeNotWanted.
        """
        if overwrite not in 'never if_broken always'.split():
            raise RuntimeError(f"Invalid 'overwrite' setting {overwrite}. ")

        self.take_only = strax.to_str_tuple(take_only)
        self.exclude = strax.to_str_tuple(exclude)
        self.overwrite = overwrite
        self.readonly = readonly
        self.log = logging.getLogger(self.__class__.__name__)

    def loader(self, key: DataKey, ambiguous='warn',
               fuzzy_for=tuple(),
               fuzzy_for_options=tuple(),
               executor=None):
        """Return loader for data described by DataKey.
        :param key: DataKey describing data
        :param ambiguous: Behaviour if multiple matching data entries are
        found:
        - 'error': Raise AmbigousDataRequest exception.
        - 'warn' (default): warn with AmbiguousDataDescription.
        - 'ignore': do nothing. Return first match.
        In the latter two cases, the first match is returned.
        :param fuzzy_for: list/tuple of plugin names for which no
        plugin name, version, or option check is performed.
        :param fuzzy_for_options: list/tuple of configuration options for which
        no check is performed.
        :param executor: Executor for pushing load computation to
        """
        backend, backend_key = self.find(key,
                                         ambiguous=ambiguous,
                                         write=False,
                                         fuzzy_for=fuzzy_for,
                                         fuzzy_for_options=fuzzy_for_options)
        return self._get_backend(backend).loader(backend_key, executor)

    def saver(self, key, metadata, meta_only):
        """Return saver for data described by DataKey."""
        backend, backend_key = self.find(key, write=True)
        return self._get_backend(backend).saver(backend_key,
                                                metadata, meta_only)

    def get_metadata(self, key,
                     ambiguous='warn',
                     fuzzy_for=tuple(),
                     fuzzy_for_options=tuple()):
        """Retrieve data-level metadata for the specified key.
        Other parameters are the same as for .find
        """
        backend, backend_key = self.find(key,
                                         write=False,
                                         ambiguous=ambiguous,
                                         fuzzy_for=fuzzy_for,
                                         fuzzy_for_options=fuzzy_for_options)
        return self._get_backend(backend).get_metadata(backend_key)

    def find(self, key: DataKey,
             write=False, ambiguous='warn',
             fuzzy_for=tuple(), fuzzy_for_options=tuple()):
        """Return (str: backend class name, backend-specific) key
        to get at / write data, or raise exception.
        :param key: DataKey of data to load
        {data_type: (plugin_name, version, {config_option: value, ...}, ...}
        :param write: Set to True if writing new data. The data is immediately
        registered, so you must follow up on the write!
        """
        if ambiguous not in 'error warn ignore'.split():
            raise RuntimeError(f"Invalid 'ambiguous' setting {ambiguous}. ")

        message = (
            f"\nRequested lineage: {key.lineage}."
            f"\nIgnoring plugin lineage for: {fuzzy_for}."
            f"\nIgnoring config options: {fuzzy_for}.")

        # Easy failures
        if (key.data_type in self.exclude
                or self.take_only and key.data_type not in self.take_only):
            raise DataNotAvailable(
                f"{self} does not accept or provide data type {key.data_type}")
        if write:
            if self.readonly:
                raise DataNotAvailable("f{self} cannot write any-data, "
                                       "it's readonly")
            try:
                at = self.find(key, write=False,
                               ambiguous=ambiguous,
                               fuzzy_for=fuzzy_for,
                               fuzzy_for_options=fuzzy_for_options)
                raise DataExistsError(
                    at=at,
                    message=(f"Data already exists at {at}.\n"
                             + message))
            except DataNotAvailable:
                pass

        try:
            return self._find(key, write,
                              fuzzy_for, fuzzy_for_options)
        except DataNotAvailable:
            raise DataNotAvailable(
                f"{key.data_type} for {key.run_id} not available." + message)
        except AmbiguousDataRequest as e:
            found = e.found
            message = (f"Found {len(found)} data entries for {key.run_id}, "
                       f"{key.data_type}: {found}." + message)
            if ambiguous == 'warn':
                warnings.warn(message)
            elif ambiguous == 'error':
                raise AmbiguousDataRequest(found=found, message=message)

    def _get_backend(self, backend):
        for b in self.backends:
            if b.__class__.__name__ == backend:
                return b
        raise KeyError(f"Unknown storage backend {backend} specified")

    def _matches(self, lineage: dict, desired_lineage: dict,
                 fuzzy_for: tuple, fuzzy_for_options: tuple):
        """Return if lineage matches desired_lineage given ignore options
        """
        if not (fuzzy_for or fuzzy_for_options):
            return lineage == desired_lineage
        args = [fuzzy_for, fuzzy_for_options]
        return (
            self._filter_lineage(lineage, *args)
            == self._filter_lineage(desired_lineage, *args))

    @staticmethod
    def _filter_lineage(lineage, fuzzy_for, fuzzy_for_options):
        """Return lineage without parts to be ignored in matching"""
        return {data_type: (v[0],
                            v[1],
                            {option_name: b
                             for option_name, b in v[2].items()
                             if option_name not in fuzzy_for_options})
                for data_type, v in lineage.items()
                if data_type not in fuzzy_for}

    def _can_overwrite(self, key):
        if self.overwrite == 'always':
            return True
        if self.overwrite == 'if_broken':
            metadata = self.get_metadata(key)
            return ('writing_ended' in metadata
                    and 'exception' not in metadata)
        return False

    ##
    # Abstract methods (to override in child)
    ##

    def _find(self, key: DataKey,
              write, fuzzy_for, fuzzy_for_options):
        """Return backend key (e.g. for filename) for data identified by key,
        raise DataNotAvailable, AmbiguousDataRequest, or DataExistsError
        Parameters are as for find.
        """
        # Use the self._matches attribute to compare lineages according to
        # the fuzzy options
        raise NotImplementedError

    def run_metadata(self, run_id):
        """Return run metadata dictionary, or raise RunMetadataNotAvailable"""
        raise NotImplementedError

    def write_run_metadata(self, run_id, metadata):
        """Stores metadata for run_id. Silently overwrites any previously
        stored run-level metadata."""
        raise NotImplementedError

    def remove(self, key):
        """Removes a registration. Does not delete any actual data"""
        raise NotImplementedError


@export
class StorageBackend:
    """Storage backend for strax data.

    This is a 'dumb' interface to data. Each bit of data stored is described
    by backend-specific keys (e.g. directory names).
    Finding and assigning backend keys is the responsibility of the
    StorageFrontend.

    The backend class name + backend_key must together uniquely identify a
    piece of data. So don't make __init__ take options like 'path' or 'host',
    these have to be hardcoded (or made part of the key).
    """
    
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def loader(self, backend_key, executor=None):
        """Iterates over strax data in backend_key
        :param executor: Executor to push load/decompress operations to
        """
        metadata = self.get_metadata(backend_key)
        if not len(metadata['chunks']):
            self.log.warning(f"No actual data in {backend_key}?")
        dtype = literal_eval(metadata['dtype'])
        compressor = metadata['compressor']

        for chunk_info in metadata['chunks']:
            kwargs = dict(chunk_info=chunk_info,
                          dtype=dtype,
                          compressor=compressor)
            if executor is None:
                yield self._read_chunk(backend_key, **kwargs)
            else:
                yield executor.submit(self._read_chunk, backend_key, **kwargs)

    def saver(self, key, metadata, meta_only=False):
        """Return saver for data described by key"""
        metadata.setdefault('compressor', 'blosc')  # TODO wrong place?
        metadata['strax_version'] = strax.__version__
        if 'dtype' in metadata:
            metadata['dtype'] = metadata['dtype'].descr.__repr__()
        return self._saver(key, metadata, meta_only=False)

    ##
    # Abstract methods (to override in child)
    ##

    def get_metadata(self, backend_key):
        """Return metadata of data described by key.
        """
        raise NotImplementedError

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        """Return a single data chunk"""
        raise NotImplementedError

    def _saver(self, key, metadata, meta_only=False):
        raise NotImplementedError


@export             # Needed for type hints elsewhere
class Saver:
    """Interface for saving a data type

    Must work even if forked.
    Do NOT add unpickleable things as attributes (such as loggers)!
    """
    closed = False
    prefer_rechunk = True

    def __init__(self, metadata, meta_only=False):
        self.meta_only = meta_only
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

    def close(self, wait_for=None, timeout=120):
        if self.closed:
            raise RuntimeError(f"{self.key.data_type} saver already closed")

        if wait_for:
            done, not_done = wait(wait_for, timeout=timeout)
            if len(not_done):
                raise RuntimeError(
                    f"{len(not_done)} futures of {self.key} did not"
                    "complete in time!")
        else:
            pass

        self.closed = True

        exc_info = sys.exc_info()
        if exc_info[0] not in [None, StopIteration]:
            self.md['exception'] = traceback.format_exc()
        self.md['writing_ended'] = time.time()

        self._close()

    ##
    # Abstract methods (to override in child)
    ##

    def _save_chunk(self, data, chunk_info):
        raise NotImplementedError

    def _save_chunk_metadata(self, chunk_info):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
