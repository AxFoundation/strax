import glob
import os
import os.path as osp
import json
import shutil

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()


run_metadata_filename = 'run_%s_metadata.json'


@export
class DataDirectory(StorageFrontend):
    """Simplest registry: single directory with FileStore data
    sitting in subdirectories.

    Run-level metadata is stored in loose json files in the directory.
    """
    def __init__(self, path='.', *args, **kwargs):
        """
        :param path: Path to folder with data subfolders.
        For other arguments, see DataRegistry base class.
        """
        super().__init__(*args, **kwargs)
        self.backends = [strax.FileSytemBackend()]
        self.path = path
        if not osp.exists(path):
            os.makedirs(path)

    def _run_meta_path(self, run_id):
        return osp.join(self.path, run_metadata_filename % run_id)

    def run_metadata(self, run_id):
        path = self._run_meta_path(run_id)
        if osp.exists(path):
            with open(path, mode='r') as f:
                return json.loads(f.read())
        else:
            raise strax.RunMetadataNotAvailable(
                f"No file at {path}, cannot find run metadata for {run_id}")

    def write_run_metadata(self, run_id, metadata):
        with open(self._run_meta_path(run_id), mode='w') as f:
            f.write(json.dumps(metadata))

    def _find(self, key,
              write, ignore_versions, ignore_config):

        # Check exact match / write case
        dirname = osp.join(self.path, str(key))
        bk = self.backend_key(dirname)
        if osp.exists(dirname):
            if write:
                if self._can_overwrite(key):
                    return bk
                raise strax.DataExistsError(at=bk)
            return bk
        if write:
            return bk

        if not ignore_versions and not ignore_config:
            raise strax.DataNotAvailable

        # Check metadata of all potentially matching data dirs for match...
        for dirname in os.listdir(self.path):
            if not osp.isdir(dirname):
                continue
            _run_id, _data_type, _ = dirname.split('_')
            if _run_id != key.run_id or _data_type != key.data_type:
                continue
            # TODO: check for broken data
            metadata = self.backends[0].get_meta(osp.join(self.path, dirname))
            if self._matches(metadata['lineage'], key.lineage,
                             ignore_versions, ignore_config):
                return self.backend_key(dirname)
        raise strax.DataNotAvailable

    def backend_key(self, dirname):
        return self.backends[0].__class__.__name__, dirname

    def remove(self, key):
        # There is no database, so removing the folder from the filesystem
        # (which FileStore should do) is sufficient.
        pass


@export
class FileSytemBackend(strax.StorageBackend):
    """Store data locally in a directory of binary files.

    Files are named after the chunk number (without extension).
    Metadata is stored in a file called metadata.json.
    """

    def get_metadata(self, dirname):
        with open(osp.join(dirname, 'metadata.json'), mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        fn = osp.join(dirname, chunk_info['filename'])
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, meta_only=False):
        # Test if the parent directory is writeable.
        # We need abspath since the dir itself may not exist,
        # even though its parent-to-be does
        parent_dir = os.path.abspath(os.path.join(dirname, os.pardir))
        if not os.access(parent_dir, os.W_OK):
            raise strax.CannotWriteData(
                f"Can't write data to {dirname}, "
                f"no write permissions in {parent_dir}.")
        return FileSaver(dirname, metadata=metadata, meta_only=meta_only)


@export
class FileSaver(strax.Saver):
    """Saves data to compressed binary files"""
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, dirname, metadata, meta_only):
        super().__init__(metadata, meta_only)
        self.dirname = dirname
        self.tempdirname = dirname + '_temp'
        if os.path.exists(dirname):
            print(f"Deleting old incomplete data in {dirname}")
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

    def _close(self):
        for fn in sorted(glob.glob(
                self.tempdirname + '/metadata_*.json')):
            with open(fn, mode='r') as f:
                self.md['chunks'].append(json.load(f))
            os.remove(fn)

        with open(self.tempdirname + '/metadata.json', mode='w') as f:
            f.write(json.dumps(self.md, **self.json_options))
        os.rename(self.tempdirname, self.dirname)
