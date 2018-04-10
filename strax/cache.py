"""Storage backends for strax.

Currently only filesystem-based storage; later probably also
data-base backed storage.
"""
from ast import literal_eval
from collections import namedtuple
import json
import logging
import os
import shutil
import time
import typing

import strax
export, __all__ = strax.exporter()

CacheKey = namedtuple('CacheKey',
                      ('run_id', 'data_type', 'lineage'))
export(CacheKey)


@export
class NotCachedException(Exception):
    pass


@export
class FileStorage:
    def __init__(self, provides='all', data_dirs=('./strax_data',)):
        """File-based storage backend for strax.

        :param wants: List of data types this backend stores.
        Defaults to 'all', accepting any data types.
        Attempting to save unwanted data types throws FileNotFoundError.
        Attempting to read unwanted data types throws NotCachedException.

        :param data_dirs: List of data directories to use.
        Entries are preferred for use from first to last. Each entry can be:
         - string: use the specified directory for all data types
         - a tuple (data_types: list/tuple/str, filename: str):
           use the directory only if the data type is in data_types.

        When writing, we save in the highest-preference directory
        in which we have write permission.
        """
        self.data_dirs = data_dirs
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

    def get(self, key: CacheKey):
        """Return generator over cached results,
        or raise NotCachedException if we have not cached the results yet
        """
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
        return self.read(dirname)

    def _candidate_dirs(self, key):
        """Iterate over directories in which key might be found"""
        if not self.provides(key.data_type):
            return

        for d in self.data_dirs:
            if isinstance(d, tuple):
                dtypes, dirname = d
                if not key.data_type in dtypes:
                    continue
            else:
                dirname = d

            # TODO: add hash of lineage to dirname?
            yield os.path.join(dirname,
                               key.run_id + '_' + key.data_type)

    def read(self, dirname: str):
        """Iterates over strax results from directory dirname"""
        metadata = JSONFileMetadata(dirname).read()
        if not len(metadata['chunks']):
            self.log.warning(f"No data files in {dirname}?")
        for chunk_info in metadata['chunks']:
            fn = os.path.join(dirname, chunk_info['filename'])
            yield strax.load(fn,
                             dtype=literal_eval(metadata['dtype']),
                             compressor=metadata['compressor'])

    def save(self, source: typing.Generator, key: CacheKey, metadata: dict):
        """Iterate over source and save the results under key
        along with metadata"""
        metadata.setdefault('compressor', 'blosc')
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

        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

        source = strax.chunk_arrays.fixed_size_chunks(source)

        with JSONFileMetadata(dirname, metadata) as md:
            for chunk_i, x in enumerate(source):
                fn = '%06d' % chunk_i

                chunk_nfo = dict(chunk_i=chunk_i,
                                  filename=fn,
                                  n=len(x),
                                  nbytes=x.nbytes)
                if 'time' in x[0].dtype.names:
                    for desc, i in (('first', 0), ('last', -1)):
                        chunk_nfo[f'{desc}_time'] = int(x[i]['time'])
                        chunk_nfo[f'{desc}_endtime'] = int(strax.endtime(x[i]))

                fn = os.path.join(dirname, fn)
                chunk_nfo['filesize'] = strax.save(fn, x, save_meta=False)
                md.add_chunk_info(chunk_nfo)


class JSONFileMetadata:

    def __init__(self, dirname, metadata=None):
        self.filename = os.path.join(dirname, 'metadata.json')
        self.md = metadata

    def read(self):
        with open(self.filename, mode='r') as f:
            return json.loads(f.read())

    def add_chunk_info(self, chunk_md):
        self.md['chunks'].append(chunk_md)
        # TODO: perhaps update? Meh, that's for a database backend

    def __enter__(self):
        self.f = open(self.filename, mode='w')
        if self.md is None:
            self.md = dict()
        self.md['chunks'] = []
        self.md['writing_started'] = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.md['writing_ended'] = time.time()
        if exc_type is not None:
            self.md['exception'] = dict(
                type=str(exc_type),
                value=str(exc_val),
                traceback=str(exc_tb)
            )

        self.f.write(json.dumps(self.md, sort_keys=True, indent=4))
        self.f.close()
