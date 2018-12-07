import glob
import json
import os
import os.path as osp
import shutil

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()


RUN_METADATA_FILENAME = 'run_%s_metadata.json'


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
        if not self.readonly and not osp.exists(self.path):
            os.makedirs(self.path)

    def _run_meta_path(self, run_id):
        return osp.join(self.path, RUN_METADATA_FILENAME % run_id)

    def run_metadata(self, run_id, projection=None):
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

    def _list_available(self, key: strax.DataKey,
                        allow_incomplete, fuzzy_for, fuzzy_for_options):
        if allow_incomplete:
            raise NotImplementedError(
                "allow_incomplete not yet supported with list_available "
                "for DataDirectory")

        found_runs = []
        for fn in self._subfolders():
            run_id = self._folder_matches(
                fn, key, fuzzy_for, fuzzy_for_options,
                ignore_name=True)
            if run_id:
                found_runs.append(run_id)

        return found_runs

    def _find(self, key, write,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        dirname = osp.join(self.path, str(key))
        exists = os.path.exists(dirname)
        bk = self.backend_key(dirname)

        if write:
            if exists and not self._can_overwrite(key):
                raise strax.DataExistsError(at=dirname)
            return bk

        if allow_incomplete:
            # Check for incomplete data (only exact matching for now)
            if fuzzy_for or fuzzy_for_options:
                raise NotImplementedError(
                    "Mixing of fuzzy matching and allow_incomplete "
                    "not supported by DataDirectory.")
            tempdirname = dirname + '_temp'
            bk = self.backend_key(tempdirname)
            if osp.exists(tempdirname):
                return bk

        # Check exact match
        if exists and self._folder_matches(dirname, key, None, None):
            return bk

        # Check metadata of all potentially matching data dirs for match...
        for fn in self._subfolders():
            if self._folder_matches(fn, key,
                                    fuzzy_for, fuzzy_for_options):
                return self.backend_key(fn)

        raise strax.DataNotAvailable

    def _subfolders(self):
        """Loop over subfolders of self.path"""
        if not os.path.exists(self.path):
            return
        for dirname in os.listdir(self.path):
            yield osp.join(self.path, dirname)

    def _folder_matches(
            self, fn, key, fuzzy_for, fuzzy_for_options,
            ignore_name=False):
        """Return the run_id of folder fn if it matches key, or False if it
        does not
        :param name: Ignore the run name part of the key. Useful for listing
        availability
        """
        # Parse the folder name
        stuff = osp.normpath(fn).split(os.sep)[-1].split('-')
        if len(stuff) != 3:
            # This is not a folder with strax data
            return False
        _run_id, _data_type, _hash = stuff

        # Check exact match
        if _data_type != key.data_type:
            return False
        if not ignore_name and _run_id != key.run_id:
            return False

        # Check fuzzy match
        if not (fuzzy_for or fuzzy_for_options):
            if _hash == key.lineage_hash:
                return _run_id
            return False
        metadata = self.backends[0].get_metadata(fn)
        if self._matches(metadata['lineage'], key.lineage,
                         fuzzy_for, fuzzy_for_options):
            return _run_id
        return False

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

    def _saver(self, dirname, metadata):
        # Test if the parent directory is writeable.
        # We need abspath since the dir itself may not exist,
        # even though its parent-to-be does
        parent_dir = os.path.abspath(os.path.join(dirname, os.pardir))

        # In case the parent dir also doesn't exist, we have to create is
        # otherwise the write permission check below will certainly fail
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except OSError as e:
            raise strax.DataNotAvailable(
                f"Can't write data to {dirname}, "
                f"{parent_dir} does not exist and we could not create it."
                f"Original error: {e}")

        # Finally, check if we have permission to create the new subdirectory
        # (which the Saver will do)
        if not os.access(parent_dir, os.W_OK):
            raise strax.DataNotAvailable(
                f"Can't write data to {dirname}, "
                f"no write permissions in {parent_dir}.")

        return FileSaver(dirname, metadata=metadata)


@export
class FileSaver(strax.Saver):
    """Saves data to compressed binary files"""
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, dirname, metadata,):
        super().__init__(metadata)
        self.dirname = dirname
        self.tempdirname = dirname + '_temp'
        if os.path.exists(dirname):
            print(f"Deleting old incomplete data in {dirname}")
            shutil.rmtree(dirname)
        if os.path.exists(self.tempdirname):
            shutil.rmtree(self.tempdirname)
        os.makedirs(self.tempdirname)
        self._flush_metadata()

    def _flush_metadata(self):
        with open(self.tempdirname + '/metadata.json', mode='w') as f:
            f.write(json.dumps(self.md, **self.json_options))

    def _save_chunk(self, data, chunk_info):
        filename = '%06d' % chunk_info['chunk_i']
        filesize = strax.save_file(
            os.path.join(self.tempdirname, filename),
            data=data,
            compressor=self.md['compressor'])
        return dict(filename=filename, filesize=filesize)

    def _save_chunk_metadata(self, chunk_info):
        if self.is_forked:
            # Write a separate metadata.json file for each chunk
            fn = f'{self.tempdirname}/metadata_{chunk_info["filename"]}.json'
            with open(fn, mode='w') as f:
                f.write(json.dumps(chunk_info, **self.json_options))
        else:
            # Just append and flush the metadata
            # (maybe not super-efficient to write the json everytime...
            # just don't use thousands of chunks)
            # TODO: maybe make option to turn this off?
            self.md['chunks'].append(chunk_info)
            self._flush_metadata()

    def _close(self):
        if not os.path.exists(self.tempdirname):
            raise RuntimeError(
                f"{self.tempdirname} was already renamed to {self.dirname}. "
                "Did you attemt to run two savers pointing to the same "
                "directory? Otherwise this could be a strange race "
                "condition or bug.")
        for fn in sorted(glob.glob(
                self.tempdirname + '/metadata_*.json')):
            with open(fn, mode='r') as f:
                self.md['chunks'].append(json.load(f))
            os.remove(fn)

        self._flush_metadata()

        os.rename(self.tempdirname, self.dirname)
