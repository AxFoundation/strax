"""Base classes for storage backends, frontends, and savers in strax.

Please see the developer documentation for more details
on strax' storage hierarchy.
"""
import logging
import time
import typing
from ast import literal_eval
from concurrent.futures import wait

import numpy as np

import strax

export, __all__ = strax.exporter()


@export
class DataKey:
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

    # Do NOT use directly, use the lineage_hash method
    _lineage_hash = ''

    def __init__(self, run_id, data_type, lineage):
        self.run_id = run_id
        self.data_type = data_type
        self.lineage = lineage

    def __repr__(self):
        return '-'.join([self.run_id, self.data_type, self.lineage_hash])

    @property
    def lineage_hash(self):
        """Deterministic hash of the lineage"""
        # We cache the hash computation to benefit tight loops calling
        # this property
        if self._lineage_hash == '':
            self._lineage_hash = strax.deterministic_hash(self.lineage)
        return strax.deterministic_hash(self.lineage)


@export
class DataNotAvailable(Exception):
    """Raised when requested data is not available"""
    pass


@export
class DataExistsError(Exception):
    """Raised when attempting to write a piece of data
    that is already written"""
    def __init__(self, at, message=''):
        super().__init__(message)
        self.at = at


@export
class DataCorrupted(Exception):
    pass


@export
class RunMetadataNotAvailable(Exception):
    pass


@export
class StorageFrontend:
    """Interface to something that knows data-locations and run-level metadata.
    For example, a runs database, or a data directory on the file system.
    """
    backends: list
    can_define_runs = False
    provide_run_metadata = False

    def __init__(self,
                 readonly=False,
                 provide_run_metadata=None,
                 overwrite='if_broken',
                 take_only=tuple(),
                 exclude=tuple()):
        """
        :param readonly: If True, throws CannotWriteData whenever saving is
        attempted.
        :param overwrite: When to overwrite data that already exists.
         - 'never': Never overwrite any data.
         - 'if_broken': Only overwrites data if it is incomplete or broken.
         - 'always': Always overwrite data. Use with caution!
        :param take_only: Provide/accept only these data types.
        :param exclude: Do NOT provide/accept these data types.
        :param provide_run_metadata: Whether to provide run-level metadata
        (run docs). If None, use class-specific default

        If take_only and exclude are both omitted, provide all data types.
        If a data type is listed in both, it will not be provided.
        Attempting to read/write unwanted data types throws DataTypeNotWanted.
        """
        if overwrite not in 'never if_broken always'.split():
            raise RuntimeError(f"Invalid 'overwrite' setting {overwrite}. ")

        self.take_only = strax.to_str_tuple(take_only)
        self.exclude = strax.to_str_tuple(exclude)
        self.overwrite = overwrite
        if provide_run_metadata is not None:
            self.provide_run_metadata = provide_run_metadata
        self.readonly = readonly
        self.log = logging.getLogger(self.__class__.__name__)

    def loader(self, key: DataKey,
               n_range=None,
               allow_incomplete=False,
               fuzzy_for=tuple(),
               fuzzy_for_options=tuple(),
               executor=None):
        """Return loader for data described by DataKey.
        :param key: DataKey describing data
        :param n_range: 2-length arraylike of (start, exclusive end)
        of row numbers to get. Default is None, which means get the entire
        run.
        :param allow_incomplete: Allow loading of data which has not been
        completely written to disk yet.
        :param fuzzy_for: list/tuple of plugin names for which no
        plugin name, version, or option check is performed.
        :param fuzzy_for_options: list/tuple of configuration options for which
        no check is performed.
        :param executor: Executor for pushing load computation to
        """
        backend, backend_key = self.find(key,
                                         write=False,
                                         allow_incomplete=allow_incomplete,
                                         fuzzy_for=fuzzy_for,
                                         fuzzy_for_options=fuzzy_for_options)
        return self._get_backend(backend).loader(
            backend_key, n_range, executor)

    def saver(self, key, metadata):
        """Return saver for data described by DataKey."""
        backend, backend_key = self.find(key, write=True)
        return self._get_backend(backend).saver(backend_key,
                                                metadata)

    def get_metadata(self, key,
                     allow_incomplete=False,
                     fuzzy_for=tuple(),
                     fuzzy_for_options=tuple()):
        """Retrieve data-level metadata for the specified key.
        Other parameters are the same as for .find
        """
        backend, backend_key = self.find(key,
                                         write=False,
                                         check_broken=False,
                                         allow_incomplete=allow_incomplete,
                                         fuzzy_for=fuzzy_for,
                                         fuzzy_for_options=fuzzy_for_options)
        return self._get_backend(backend).get_metadata(backend_key)

    def _we_take(self, data_type):
        """Return if data_type can be provided by this frontend"""
        return not (data_type in self.exclude
                    or self.take_only and data_type not in self.take_only)

    def find(self, key: DataKey,
             write=False,
             check_broken=True,
             allow_incomplete=False,
             fuzzy_for=tuple(), fuzzy_for_options=tuple()):
        """Return (str: backend class name, backend-specific) key
        to get at / write data, or raise exception.
        :param key: DataKey of data to load
        {data_type: (plugin_name, version, {config_option: value, ...}, ...}
        :param write: Set to True if writing new data. The data is immediately
        registered, so you must follow up on the write!
        :param check_broken: If True, raise DataNotAvailable if data has not
        been complete written, or writing terminated with an exception.
        """
        message = (
            f"\nRequested lineage: {key.lineage}."
            f"\nIgnoring plugin lineage for: {fuzzy_for}."
            f"\nIgnoring config options: {fuzzy_for}.")

        if not self._we_take(key.data_type):
            raise DataNotAvailable(
                f"{self} does not accept or provide data type {key.data_type}")

        if write:
            if self.readonly:
                raise DataNotAvailable(f"{self} cannot write any-data, "
                                       "it's readonly")
            try:
                at = self.find(key, write=False,
                               allow_incomplete=allow_incomplete,
                               fuzzy_for=fuzzy_for,
                               fuzzy_for_options=fuzzy_for_options)
                raise DataExistsError(
                    at=at,
                    message=(f"Data already exists at {at}.\n"
                             + message))
            except DataNotAvailable:
                pass

        try:
            backend_name, backend_key = self._find(
                key=key,
                write=write,
                allow_incomplete=allow_incomplete,
                fuzzy_for=fuzzy_for,
                fuzzy_for_options=fuzzy_for_options)
        except DataNotAvailable:
            raise DataNotAvailable(
                f"{key.data_type} for {key.run_id} not available." + message)

        if not write and check_broken:
            # Get the metadata to check if the data is broken
            meta = self._get_backend(backend_name).get_metadata(backend_key)
            if 'exception' in meta:
                exc = meta['exception']
                raise DataNotAvailable(
                    f"Data in {backend_name} {backend_key} corrupted due to "
                    f"exception during writing: {exc}.")
            if 'writing_ended' not in meta and not allow_incomplete:
                raise DataNotAvailable(
                    f"Data in {backend_name} {backend_key} corrupted. No "
                    f"writing_ended field present!")

        return backend_name, backend_key

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

    def _can_overwrite(self, key: DataKey):
        if self.overwrite == 'always':
            return True
        if self.overwrite == 'if_broken':
            metadata = self.get_metadata(key)
            return not  ('writing_ended' in metadata
                         and 'exception' not in metadata)
        return False

    def find_several(self, keys, **kwargs):
        """Return list with backend keys or False
        for several data keys.

        Options are as for find()
        """
        # You can override this if the backend has a smarter way
        # of checking availability (e.g. a single DB query)
        result = []
        for key in keys:
            try:
                r = self.find(key, **kwargs)
            except (strax.DataNotAvailable,
                    strax.DataCorrupted):
                r = False
            result.append(r)
        return result

    ##
    # Abstract methods (to override in child)
    ##

    def _scan_runs(self, store_fields):
        """Iterable of run document / metadata dictionaries
        """
        yield from tuple()

    def _find(self, key: DataKey,
              write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        """Return backend key (e.g. for filename) for data identified by key,
        raise DataNotAvailable, or DataExistsError
        Parameters are as for find.
        """
        # Use the self._matches attribute to compare lineages according to
        # the fuzzy options
        raise NotImplementedError

    def run_metadata(self, run_id, projection=None):
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

    def loader(self, backend_key, n_range=None, executor=None):
        """Iterates over strax data in backend_key
        :param n_range: 2-length arraylike of (start, exclusive end)
        of row numbers to get. Default is None, which means get the entire
        run.
        :param executor: Executor to push load/decompress operations to
        """
        metadata = self.get_metadata(backend_key)
        if not len(metadata['chunks']):
            raise DataCorrupted(f"No chunks of data in {backend_key}")
        dtype = literal_eval(metadata['dtype'])
        compressor = metadata['compressor']

        first_row_in_chunk = np.array([c['n']
                                       for c in metadata['chunks']]).cumsum()
        first_row_in_chunk -= first_row_in_chunk[0]

        for i, chunk_info in enumerate(metadata['chunks']):
            if (n_range
                    and not n_range[0] <= first_row_in_chunk[i] < n_range[1]):
                continue
            kwargs = dict(chunk_info=chunk_info,
                          dtype=dtype,
                          compressor=compressor)
            if executor is None:
                yield self._read_chunk(backend_key, **kwargs)
            else:
                yield executor.submit(self._read_chunk, backend_key, **kwargs)

    def saver(self, key, metadata):
        """Return saver for data described by key"""
        metadata.setdefault('compressor', 'blosc')  # TODO wrong place?
        metadata['strax_version'] = strax.__version__
        if 'dtype' in metadata:
            metadata['dtype'] = metadata['dtype'].descr.__repr__()
        return self._saver(key, metadata)

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

    def _saver(self, key, metadata):
        raise NotImplementedError


@export             # Needed for type hints elsewhere
class Saver:
    """Interface for saving a data type

    Must work even if forked.
    Do NOT add unpickleable things as attributes (such as loggers)!
    """
    closed = False
    allow_rechunk = True   # If False, do not rechunk even if plugin allows it
    allow_fork = True   # If False, cannot be inlined / forked

    # This is set if the saver is operating in multiple processes at once
    # Do not set it yourself
    is_forked = False

    got_exception = None

    def __init__(self, metadata):
        self.md = metadata
        self.md['writing_started'] = time.time()
        self.md['chunks'] = []

    def save_from(self, source: typing.Iterable, rechunk=True, executor=None):
        """Iterate over source and save the results under key
        along with metadata
        """
        if rechunk and self.allow_rechunk:
            source = strax.fixed_size_chunks(source)

        pending = []
        try:
            for chunk_i, s in enumerate(source):
                new_f = self.save(data=s, chunk_i=chunk_i, executor=executor)
                if new_f is not None:
                    pending = [f for f in pending + [new_f]
                               if not f.done()]

        except strax.MailboxKilled:
            # Write exception (with close), but exit gracefully.
            # One traceback on screen is enough
            self.close(wait_for=pending)
            pass

        except Exception as e:
            # log exception for the final check
            self.got_exception = e
            # Throw the exception back into the mailbox
            # (hoping that it is still listening...)
            source.throw(e)
            raise e

        finally:
            if not self.closed:
                self.close(wait_for=pending)

    def save(self, data: np.ndarray, chunk_i: int, executor=None):
        """Save a chunk, returning future to wait on or None"""
        if self.closed:
            raise RuntimeError(f"Attmpt to save to {self.md} saver, "
                               f"which is already closed!")

        chunk_info = dict(chunk_i=chunk_i,
                          n=len(data),
                          nbytes=data.nbytes)
        if len(data) != 0 and 'time' in data[0].dtype.names:
            for desc, i in (('first', 0), ('last', -1)):
                chunk_info[f'{desc}_time'] = int(data[i]['time'])
                chunk_info[f'{desc}_endtime'] = int(strax.endtime(data[i]))

        bonus_info, future = self._save_chunk(
            data,
            chunk_info,
            executor=None if self.is_forked else executor)

        chunk_info.update(bonus_info)
        self._save_chunk_metadata(chunk_info)

        return future

    def close(self,
              wait_for: typing.Union[list, tuple] = tuple(),
              timeout=300):
        if self.closed:
            raise RuntimeError(f"{self.md} saver already closed")

        if wait_for:
            done, not_done = wait(wait_for, timeout=timeout)
            if len(not_done):
                raise RuntimeError(
                    f"{len(not_done)} futures of {self.md} did not"
                    "complete in time!")

        self.closed = True

        exc_info = strax.formatted_exception()
        if exc_info:
            self.md['exception'] = exc_info

        self.md['writing_ended'] = time.time()

        self._close()

    ##
    # Abstract methods (to override in child)
    ##

    def _save_chunk(self, data, chunk_info, executor=None):
        """Save a chunk to file. Return (
            dict with extra info for metadata,
            future to wait on or None)
        """
        raise NotImplementedError

    def _save_chunk_metadata(self, chunk_info):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError
