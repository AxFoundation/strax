import glob
import json
import tempfile
import os
import os.path as osp

from bson import json_util
import shutil

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()


RUN_METADATA_PATTERN = '%s-metadata.json'


@export
class DataDirectory(StorageFrontend):
    """Simplest registry: single directory with FileStore data
    sitting in subdirectories.

    Run-level metadata is stored in loose json files in the directory.
    """

    provide_run_metadata = True

    def __init__(self, path='.', *args, deep_scan=True, **kwargs):
        """
        :param path: Path to folder with data subfolders.
        :param deep_scan: Let scan_runs scan over folders,
        so even data for which no run-level metadata is available
        is reported.

        For other arguments, see DataRegistry base class.
        """
        super().__init__(*args, **kwargs)
        self.backends = [strax.FileSytemBackend()]
        self.path = path
        self.deep_scan = deep_scan
        if not self.readonly and not osp.exists(self.path):
            os.makedirs(self.path)

    def _run_meta_path(self, run_id):
        return osp.join(self.path, RUN_METADATA_PATTERN % run_id)

    def run_metadata(self, run_id, projection=None):
        path = self._run_meta_path(run_id)
        if osp.exists(path):
            with open(path, mode='r') as f:
                md = json.loads(f.read(),
                                object_hook=json_util.object_hook)
            if not projection:
                return md
            md = strax.flatten_dict(md, separator='.')
            return {k: v
                    for k, v in md.items()
                    if k in projection}
        else:
            raise strax.RunMetadataNotAvailable(
                f"No file at {path}, cannot find run metadata for {run_id}")

    def write_run_metadata(self, run_id, metadata):
        with open(self._run_meta_path(run_id), mode='w') as f:
            f.write(json.dumps(metadata, default=json_util.default))

    def _scan_runs(self, store_fields):
        """Iterable of run document dictionaries.
        These should be directly convertable to a pandas DataFrame.
        """
        found = set()

        # Yield metadata for runs for which we actually have it
        for md_path in sorted(glob.glob(
                osp.join(self.path,
                         RUN_METADATA_PATTERN.replace('%s', '*')))):
            # Parse the run metadata filename pattern.
            # (different from the folder pattern)
            run_id = osp.basename(md_path).split('-')[0]
            found.add(run_id)
            yield self.run_metadata(run_id, projection=store_fields)

        if self.deep_scan:
            # Yield runs for which no metadata exists
            # we'll make "metadata" that consist only of the run name
            for fn in self._subfolders():
                run_id = self._parse_folder_name(fn)[0]
                if run_id not in found:
                    found.add(run_id)
                    yield dict(name=run_id)

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
        """Loop over subfolders of self.path that match our folder format"""
        if not os.path.exists(self.path):
            return
        for dirname in os.listdir(self.path):
            try:
                self._parse_folder_name(dirname)
            except InvalidFolderNameFormat:
                continue
            yield osp.join(self.path, dirname)

    @staticmethod
    def _parse_folder_name(fn):
        """Return (run_id, data_type, hash) if folder name matches
        DataDirectory convention, raise InvalidFolderNameFormat otherwise
        """
        stuff = osp.normpath(fn).split(os.sep)[-1].split('-')
        if len(stuff) != 3:
            # This is not a folder with strax data
            raise InvalidFolderNameFormat(fn)
        return stuff

    def _folder_matches(
            self, fn, key, fuzzy_for, fuzzy_for_options,
            ignore_name=False):
        """Return the run_id of folder fn if it matches key, or False if it
        does not
        :param name: Ignore the run name part of the key. Useful for listing
        availability
        """
        # Parse the folder name
        try:
            _run_id, _data_type, _hash = self._parse_folder_name(fn)
        except InvalidFolderNameFormat:
            return False

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
def dirname_to_prefix(dirname):
    """Return filename prefix from dirname"""
    return os.path.basename(dirname.strip('/')).split("-", maxsplit=1)[1]


@export
class FileSytemBackend(strax.StorageBackend):
    """Store data locally in a directory of binary files.

    Files are named after the chunk number (without extension).
    Metadata is stored in a file called metadata.json.
    """

    def get_metadata(self, dirname):
        prefix = dirname_to_prefix(dirname)
        metadata_json = f'{prefix}-metadata.json'
        md_path = osp.join(dirname, metadata_json)

        if not osp.exists(md_path):
            # Try old-format metadata
            # (if it's not there, just let it raise FileNotFound
            # with the usual message in the next stage)
            old_md_path = osp.join(dirname, 'metadata.json')
            if not osp.exists(old_md_path):
                raise strax.DataCorrupted(f"Data in {dirname} has no metadata")
            md_path = old_md_path

        with open(md_path, mode='r') as f:
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

    def __init__(self, dirname, metadata):
        super().__init__(metadata)
        self.dirname = dirname
        self.tempdirname = dirname + '_temp'
        self.prefix = dirname_to_prefix(dirname)
        self.metadata_json = f'{self.prefix}-metadata.json'

        if os.path.exists(dirname):
            print(f"Removing data in {dirname} to overwrite")
            shutil.rmtree(dirname)
        if os.path.exists(self.tempdirname):
            print(f"Removing old incomplete data in {dirname}")
            shutil.rmtree(self.tempdirname)
        os.makedirs(self.tempdirname)
        self._flush_metadata()

    def _flush_metadata(self):
        with open(self.tempdirname + '/' + self.metadata_json, mode='w') as f:
            f.write(json.dumps(self.md, **self.json_options))

    def _save_chunk(self, data, chunk_info, executor=None):
        ichunk = '%06d' % chunk_info['chunk_i']
        filename = f'{self.prefix}-{ichunk}'

        fn = os.path.join(self.tempdirname, filename)
        kwargs = dict(data=data, compressor=self.md['compressor'])
        if executor is None:
            filesize = strax.save_file(fn, **kwargs)
            return dict(filename=filename, filesize=filesize), None
        else:
            return dict(filename=filename), executor.submit(
                strax.save_file, fn, **kwargs)

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


@export
class InvalidFolderNameFormat(Exception):
    pass