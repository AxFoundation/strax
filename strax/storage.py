"""Storage backends for strax.

Currently only filesystem-based storage; later probably also
database backed storage.
"""
from ast import literal_eval
import concurrent.futures
import json
import logging
import os
import shutil
import traceback
import time
import typing

import strax
export, __all__ = strax.exporter()


@export
class CacheKey(typing.NamedTuple):
    run_id: str
    data_type: str
    lineage: dict


@export
class NotCachedException(Exception):
    pass


@export
class FileStorage:
    def __init__(self, provides='all', data_dirs=('./strax_data',),
                 executor=None):
        """File-based storage backend for strax.

        :param provides: List of data types this backend stores.
        Defaults to 'all', accepting any data types.
        Attempting to save unwanted data types throws FileNotFoundError.
        Attempting to read unwanted data types throws NotCachedException.

        :param data_dirs: List of data directories to use.
        Entries are preferred for use from first to last. Each entry can be:
         - string: use the specified directory for all data types
         - a tuple (data_types: list/tuple, filename: str):
           use the directory only if the data type is in data_types.

        When writing, we save in the highest-preference directory
        in which we have write permission.

        :param executor: concurrent.futures executor for parallelizing result
        computations.
        """
        self.data_dirs = data_dirs
        self.executor = executor
        self._provides = provides
        self.log = logging.getLogger(self.__class__.__name__)
        for d in data_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass
            else:
                self.log.debug(f"Created data dir {d}")

    def provides(self, data_type):
        """Return whether this backend will store this datatype"""
        if self._provides == 'all':
            return True
        if self._provides == 'none':
            return False
        return data_type in self._provides

    def has(self, key: CacheKey):
        try:
            self._find(key)
        except NotCachedException:
            return False
        return False

    def loader(self, key: CacheKey):
        """Return generator over cached results,
        or raise NotCachedException if we have not cached the results yet
        """
        return self.read(self._find(key))

    def _find(self, key):
        if not self.provides(key.data_type):
            self.log.debug(f"{key.data_type} not wanted by storage backend.")
            raise NotCachedException

        for dirname in self._candidate_dirs(key):
            if os.path.exists(dirname):
                break
        else:
            self.log.debug(f"{key} is NOT in cache.")
            raise NotCachedException
        self.log.debug(f"{key} is in cache.")

        return dirname

    def _candidate_dirs(self, key):
        """Iterate over directories in which key might be found"""
        if not self.provides(key.data_type):
            return

        for d in self.data_dirs:
            if isinstance(d, tuple):
                dtypes, dirname = d
                if key.data_type not in dtypes:
                    continue
            else:
                dirname = d

            # TODO: add hash of lineage to dirname?
            yield os.path.join(dirname,
                               key.run_id + '_' + key.data_type)

    def read(self, dirname: str):
        """Iterates over strax results from directory dirname"""
        with open(dirname + '/metadata.json', mode='r') as f:
            metadata = json.loads(f.read())
        if not len(metadata['chunks']):
            self.log.warning(f"No data files in {dirname}?")
        dtype = literal_eval(metadata['dtype'])
        compressor = metadata['compressor']

        kwargs = dict(dtype=dtype, compressor=compressor)
        for chunk_info in metadata['chunks']:
            fn = os.path.join(dirname, chunk_info['filename'])
            if self.executor is None:
                yield strax.load_file(fn, **kwargs)
            else:
                yield self.executor.submit(strax.load_file, fn, **kwargs)

    def saver(self, key, metadata):
        metadata.setdefault('compressor', 'blosc')
        metadata['strax_version'] = strax.__version__
        if 'dtype' in metadata:
            metadata['dtype'] = metadata['dtype'].descr.__repr__()

        for dirname in self._candidate_dirs(key):
            # Test if the parent directory is writeable.
            # We need abspath since the dir itself may not exist,
            # even though its parent-to-be does
            parent_dir = os.path.abspath(os.path.join(dirname, os.pardir))
            if os.access(parent_dir, os.W_OK):
                self.log.debug(f"Saving {key} to {dirname}")
                break
            else:
                self.log.debug(f"{parent_dir} is not writeable, "
                               f"can't save to {dirname}")
        else:
            raise FileNotFoundError(f"No writeable directory found for {key}")

        return FileSaver(key, metadata, dirname, executor=self.executor)


@export
class FileSaver:
    closed = False

    def __init__(self, key, metadata, dirname, executor):
        self.key = key
        self.md = metadata
        self.dirname = dirname
        self.executor = executor
        self.futures = []

        self.tempdirname = dirname + '_temp'
        if os.path.exists(dirname):
            # TODO: Hmm.... shouldn't it already exist? No, simple_chain...
            shutil.rmtree(dirname)
        if os.path.exists(self.tempdirname):
            shutil.rmtree(self.tempdirname)
        os.makedirs(self.tempdirname)

        self.md['chunks'] = []
        self.md['writing_started'] = time.time()

    def save_from(self, source: typing.Iterable, rechunk=True):
        """Iterate over source and save the results under key
        along with metadata
        """
        if rechunk:
            source = strax.fixed_size_chunks(source)

        try:
            for chunk_i, s in enumerate(source):
                self.send(chunk_i=chunk_i, data=s)
        except Exception:
            self.crash_close()
            raise
        # TODO: should we catch MailboxKilled?
        self.close()

    def send(self, chunk_i, data):
        if self.closed:
            raise RuntimeError("Already closed!")
        fn = '%06d' % chunk_i

        n_missing = (chunk_i - len(self.md['chunks']) + 1)
        if n_missing > 0:
            self.md['chunks'] += [None] * n_missing
        assert len(self.md['chunks']) >= chunk_i + 1

        chunk_info = dict(chunk_i=chunk_i,
                          filename=fn,
                          n=len(data),
                          nbytes=data.nbytes)
        if 'time' in data[0].dtype.names:
            for desc, i in (('first', 0), ('last', -1)):
                chunk_info[f'{desc}_time'] = int(data[i]['time'])
                chunk_info[f'{desc}_endtime'] = int(strax.endtime(data[i]))

        fn = os.path.join(self.tempdirname, fn)

        kwargs = dict(filename=fn,
                      data=data,
                      compressor=self.md['compressor'])
        if self.executor is None:
            chunk_info['filesize'] = strax.save_file(**kwargs)
            self.md['chunks'][chunk_i] = chunk_info
        else:
            self.md['chunks'][chunk_i] = chunk_info
            f = self.executor.submit(strax.save_file, **kwargs)
            self.futures.append(f)

            def set_filesize(_f):
                self.md['chunks'][chunk_i]['filesize'] = _f.result()
            f.add_done_callback(set_filesize)

    def close(self, wait=True):
        if self.closed:
            raise RuntimeError("Already closed!")
        self.closed = True
        if wait:
            concurrent.futures.wait(self.futures)
        self.md['writing_ended'] = time.time()
        with open(self.tempdirname + '/metadata.json', mode='w') as f:
            f.write(json.dumps(self.md, sort_keys=True, indent=4))
        os.rename(self.tempdirname, self.dirname)

    def crash_close(self):
        self.md['exception'] = traceback.format_exc()
        self.close(wait=False)
