"""Base classes for storage backends, frontends, and savers in strax.

Please see the developer documentation for more details
on strax' storage hierarchy.
"""
from ast import literal_eval
from concurrent.futures import wait
import logging
from packaging import version
import time
import typing
import warnings

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
        return self._lineage_hash


@export
class DataNotAvailable(Exception):
    """Raised when requested data is not available"""
    pass


@export
class EmptyDataWarning(UserWarning):
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

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # List the relevant attributes ('path' is actually for the
        # strax.DataDirectory but it makes more sense to put it here).
        attributes = ('readonly', 'path', 'exclude', 'take_only')
        representation = f'{self.__class__.__module__}.{self.__class__.__name__}'
        for attr in attributes:
            if hasattr(self, attr) and getattr(self, attr):
                representation += f', {attr}: {getattr(self, attr)}'
        return representation

    def loader(self, key: DataKey,
               time_range=None,
               allow_incomplete=False,
               fuzzy_for=tuple(),
               fuzzy_for_options=tuple(),
               chunk_number=None,
               executor=None):
        """Return loader for data described by DataKey.
        :param key: DataKey describing data
        :param time_range: 2-length arraylike of (start, exclusive end)
        of row numbers to get. Default is None, which means get the entire
        run.
        :param allow_incomplete: Allow loading of data which has not been
        completely written to disk yet.
        :param fuzzy_for: list/tuple of plugin names for which no
        plugin name, version, or option check is performed.
        :param fuzzy_for_options: list/tuple of configuration options for which
        no check is performed.
        :param chunk_number: Chunk number to load exclusively.
        :param executor: Executor for pushing load computation to
        """
        backend, backend_key = self.find(key,
                                         write=False,
                                         allow_incomplete=allow_incomplete,
                                         fuzzy_for=fuzzy_for,
                                         fuzzy_for_options=fuzzy_for_options)
        return self._get_backend(backend).loader(
            backend_key,
            time_range=time_range,
            executor=executor,
            chunk_number=chunk_number)

    def saver(self, key, metadata, **kwargs):
        """Return saver for data described by DataKey."""
        backend, backend_key = self.find(key, write=True)
        return self._get_backend(backend).saver(backend_key,
                                                metadata, **kwargs)

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

    def define_run(self, name, sub_run_spec, **metadata):
        self.write_run_metadata(name, dict(
            sub_run_spec=sub_run_spec,
            **metadata))

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

    def loader(self,
               backend_key,
               time_range=None,
               chunk_number=None,
               executor=None):
        """Iterates over strax data in backend_key
        :param time_range: 2-length arraylike of (start, exclusive end)
        of desired data. Will return all data that partially overlaps with
        the range.
        Default is None, which means get the entire
        :param chunk_number: Chunk number to get exclusively
        :param executor: Executor to push load/decompress operations to
        """
        metadata = self.get_metadata(backend_key)

        if 'strax_version' in metadata:
            v_old = metadata['strax_version']
            if version.parse(v_old) < version.parse('0.9.0'):
                raise strax.DataNotAvailable(
                    f"Cannot load data at {backend_key}: "
                    f"it was created with strax {v_old}, "
                    f"but you have strax {strax.__version__}. ")
        else:
            warnings.warn(f"Data at {backend_key} does not say what strax "
                          "version it was generated with. This means it is "
                          "corrupted, or very, very old. Probably "
                          "we cannot load this.")

        # 'start' and 'end' are not required, to allow allow_incomplete
        required_fields = (
            'run_id data_type data_kind dtype compressor').split()
        missing_fields = [x for x in required_fields if x not in metadata]
        if len(missing_fields):
            raise strax.DataNotAvailable(
                f"Cannot load data at {backend_key}: metadata is "
                f"missing the required fields {missing_fields}. ")

        if not len(metadata['chunks']):
            raise ValueError(
                f"Cannot load data at {backend_key}, it has no chunks!")

        dtype = literal_eval(metadata['dtype'])

        # Common arguments for chunk construction, not stored with chunk-level
        # metadata
        chunk_kwargs = dict(
            data_type=metadata['data_type'],
            data_kind=metadata['data_kind'],
            dtype=dtype,
            target_size_mb=metadata.get('chunk_target_size_mb',
                                        strax.default_chunk_size_mb))

        required_chunk_metadata_fields = 'start end run_id'.split()

        for i, chunk_info in enumerate(strax.iter_chunk_meta(metadata)):

            missing_fields = [x for x in required_chunk_metadata_fields
                              if x not in chunk_info]
            if len(missing_fields):
                raise ValueError(
                    f"Error reading chunk {i} of {metadata['dtype']} " 
                    f"of {metadata['run_d']} from {backend_key}: "
                    f"chunk metadata is missing fields {missing_fields}")

            # Chunk number constraint
            if chunk_number is not None:
                if i != chunk_number:
                    continue

            # Time constraint
            if time_range:
                if (chunk_info['end'] <= time_range[0]
                        or time_range[1] <= chunk_info['start']):
                    # Chunk does not cover any part of range
                    continue

            read_chunk_kwargs = dict(
                backend_key=backend_key,
                dtype=dtype,
                metadata=metadata,
                chunk_info=chunk_info,
                time_range=time_range,
                chunk_construction_kwargs=chunk_kwargs)
            if executor is None:
                yield self._read_and_format_chunk(**read_chunk_kwargs)
            else:
                yield executor.submit(self._read_and_format_chunk,
                                      **read_chunk_kwargs)

    def _read_and_format_chunk(self,
                               *,
                               backend_key,
                               dtype,
                               metadata,
                               chunk_info,
                               time_range,
                               chunk_construction_kwargs) -> strax.Chunk:

        if chunk_info['n'] == 0:
            # No data, no need to load
            data = np.empty(0, dtype=dtype)
        else:
            data = self._read_chunk(backend_key,
                                    chunk_info=chunk_info,
                                    dtype=dtype,
                                    compressor=metadata['compressor'])

        result = strax.Chunk(
            start=chunk_info['start'],
            end=chunk_info['end'],
            run_id=chunk_info['run_id'],
            data=data,
            **chunk_construction_kwargs)

        if time_range:
            if result.start < time_range[0]:
                _, result = result.split(t=time_range[0],
                                         allow_early_split=True)
            if result.end > time_range[1]:
                try:
                    result, _ = result.split(t=time_range[1],
                                             allow_early_split=False)
                except strax.CannotSplit:
                    pass

        return result

    def saver(self, key, metadata, **kwargs):
        """Return saver for data described by key"""
        metadata.setdefault('compressor', 'blosc')  # TODO wrong place?
        metadata['strax_version'] = strax.__version__
        if 'dtype' in metadata:
            metadata['dtype'] = metadata['dtype'].descr.__repr__()
        return self._saver(key, metadata, **kwargs)

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

    def __init__(self, metadata, saver_timeout=300):
        self.md = metadata
        self.md['writing_started'] = time.time()
        self.md['chunks'] = []
        self.timeout = saver_timeout

    def save_from(self, source: typing.Generator, rechunk=True, executor=None):
        """Iterate over source and save the results under key
        along with metadata
        """
        pending = []
        exhausted = False
        chunk_i = 0
        try:
            while not exhausted:
                chunk = None
                try:
                    if rechunk and self.allow_rechunk:
                        while (chunk is None or
                                chunk.data.nbytes < chunk.target_size_mb*1e6):
                            chunk = strax.Chunk.concatenate(
                                [chunk, next(source)])
                    else:
                        chunk = next(source)
                except StopIteration:
                    exhausted = True

                if chunk is None:
                    break

                new_f = self.save(chunk=chunk,
                                  chunk_i=chunk_i, executor=executor)
                pending = [f for f in pending if not f.done()]
                if new_f is not None:
                    pending += [new_f]
                chunk_i += 1

        except strax.MailboxKilled:
            # Write exception (with close), but exit gracefully.
            # One traceback on screen is enough
            self.close(wait_for=pending)

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

    def save(self, chunk: strax.Chunk, chunk_i: int, executor=None):
        """Save a chunk, returning future to wait on or None"""
        if self.closed:
            raise RuntimeError(f"Attmpt to save to {self.md} saver, "
                               f"which is already closed!")

        chunk_info = dict(chunk_i=chunk_i,
                          n=len(chunk),
                          start=chunk.start,
                          end=chunk.end,
                          run_id=chunk.run_id,
                          nbytes=chunk.nbytes)
        if len(chunk) != 0 and 'time' in chunk.dtype.names:
            for desc, i in (('first', 0), ('last', -1)):
                chunk_info[f'{desc}_time'] = \
                    int(chunk.data[i]['time'])
                chunk_info[f'{desc}_endtime'] = \
                    int(strax.endtime(chunk.data[i]))

        if len(chunk):
            bonus_info, future = self._save_chunk(
                chunk.data,
                chunk_info,
                executor=None if self.is_forked else executor)
            chunk_info.update(bonus_info)
        else:
            # No need to create an empty file for an empty chunk;
            # the annotation in the metadata is sufficient.
            future = None

        self._save_chunk_metadata(chunk_info)
        return future

    def close(self,
              wait_for: typing.Union[list, tuple] = tuple()):
        if self.closed:
            raise RuntimeError(f"{self.md} saver already closed")

        if wait_for:
            done, not_done = wait(wait_for, timeout=self.timeout)
            if len(not_done):
                raise RuntimeError(
                    f"{len(not_done)} futures of {self.md} did not"
                    "complete in time!")

        self.closed = True

        exc_info = strax.formatted_exception()
        if exc_info:
            self.md['exception'] = exc_info

        if self.md['chunks']:
            # Update to precise start and end values
            self.md['start'] = self.md['chunks'][0]['start']
            self.md['end'] = self.md['chunks'][-1]['end']
        # If there were no chunks, we are certainly crashing.
        # Don't throw another exception

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
