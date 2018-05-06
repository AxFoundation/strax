import os
import json
import shutil
import glob

import strax
from .common import Store, Saver, NotCached, CacheKey

export, __all__ = strax.exporter()


@export
class FileStore(Store):

    def __init__(self, data_dirs, *args, **kwargs):
        """File-based storage backend for strax.
        Data is stored in (compressed) binary dumps, metadata in json.

        :param data_dirs: List of data directories to use.
        Entries are preferred for use from first to last.

        {provides_doc}

        When writing, we save (only) in the highest-preference directory
        in which we have write permission.
        """
        super().__init__(*args, **kwargs)

        self.data_dirs = strax.to_str_tuple(data_dirs)

        for d in self.data_dirs:
            try:
                os.makedirs(d)
            except FileExistsError:
                pass
            else:
                self.log.debug(f"Created data dir {d}")

    def _find(self, key):
        for dirname in self._candidate_dirs(key):
            if os.path.exists(dirname):
                self.log.debug(f"{key} is in cache.")
                return dirname
        self.log.debug(f"{key} is NOT in cache.")
        raise NotCached

    @staticmethod
    def _key_dirname(key):
        return '_'.join([key.run_id,
                         key.data_type,
                         strax.deterministic_hash(key.lineage)])

    def _candidate_dirs(self, key: CacheKey):
        """Return directories at which data meant by key
        could be found or saved"""
        return [os.path.join(d, str(key))
                for d in self.data_dirs]

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        fn = os.path.join(dirname, chunk_info['filename'])
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _read_meta(self, dirname):
        with open(dirname + '/metadata.json', mode='r') as f:
            return json.loads(f.read())

    def saver(self, key, metadata):
        super().saver(key, metadata)

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

        return FileSaver(key, metadata, dirname)


@export
class FileSaver(Saver):
    """Saves data to compressed binary files

    Must work even if forked.
    Do NOT add unpickleable things as attributes (such as loggers)!
    """
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, key, metadata, dirname):
        super().__init__(key, metadata)
        self.dirname = dirname

        self.tempdirname = dirname + '_temp'
        if os.path.exists(dirname):
            print("Deleting old incomplete data in {dirname}")
            shutil.rmtree(dirname)
        if os.path.exists(self.tempdirname):
            shutil.rmtree(self.tempdirname)
        os.makedirs(self.tempdirname)

    def _save_chunk(self, data, chunk_info):
        filename = '%06d' % chunk_info['chunk_i']
        filesize = strax.save_file(
                os.path.join(self.tempdirname, filename),
                data=data,
                compressor=self.md['compressor'])
        return dict(filename=filename, filesize=filesize)

    def _save_chunk_metadata(self, chunk_info):
        if self.meta_only:
            # TODO HACK!
            chunk_info["filename"] = '%06d' % chunk_info['chunk_i']
        fn = f'{self.tempdirname}/metadata_{chunk_info["filename"]}.json'
        with open(fn, mode='w') as f:
            f.write(json.dumps(chunk_info, **self.json_options))

    def close(self):
        super().close()
        for fn in sorted(glob.glob(
                self.tempdirname + '/metadata_*.json')):
            with open(fn, mode='r') as f:
                self.md['chunks'].append(json.load(f))
            os.remove(fn)

        with open(self.tempdirname + '/metadata.json', mode='w') as f:
            f.write(json.dumps(self.md, **self.json_options))
        os.rename(self.tempdirname, self.dirname)
