import glob
import json
import tempfile
import os
import os.path as osp
import typing
import time
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

    can_define_runs = True
    provide_run_metadata = False
    provide_superruns = True

    def __init__(self, path='.', *args, deep_scan=False, **kwargs):
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
        if not osp.exists(path):
            raise strax.RunMetadataNotAvailable(
                f"No file at {path}, cannot find run metadata for {run_id}")
        with open(path, mode='r') as f:
            md = json.loads(f.read(),
                            object_hook=json_util.object_hook)
        md = strax.flatten_run_metadata(md)
        if projection is not None:
            md = {k: v
                  for k, v in md.items()
                  if k in projection}
        return md

    def write_run_metadata(self, run_id, metadata):
        with open(self._run_meta_path(run_id), mode='w') as f:
            if not 'name' in metadata:
                metadata['name'] = run_id
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
        self.raise_if_non_compatible_run_id(key.run_id)
        dirname = osp.join(self.path, str(key))
        exists = os.path.exists(dirname)
        bk = self.backend_key(dirname)

        if write:
            if exists and not self._can_overwrite(key):
                raise strax.DataExistsError(at=dirname)
            return bk

        if allow_incomplete and not exists:
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

        # Check metadata of all potentially matching data dirs for
        # matches. This only makes sense for fuzzy searches since
        # otherwise we should have had an exact match already. (Also
        # really slows down st.select runs otherwise because we doing an
        # entire search over all the files in self._subfolders for all
        # non-available keys).
        if fuzzy_for or fuzzy_for_options:
            for fn in self._subfolders():
                if self._folder_matches(fn, key, fuzzy_for, fuzzy_for_options):
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

    @staticmethod
    def raise_if_non_compatible_run_id(run_id):
        if '-' in str(run_id):
            raise ValueError("The filesystem frontend does not understand"
                             " run_id's with '-', please replace with '_'")

@export
def dirname_to_prefix(dirname):
    """Return filename prefix from dirname"""
    dirname = dirname.replace('_temp', '')
    return os.path.basename(dirname.strip('/').rstrip('\\')).split("-", maxsplit=1)[1]


@export
class FileSytemBackend(strax.StorageBackend):
    """Store data locally in a directory of binary files.

    Files are named after the chunk number (without extension).
    Metadata is stored in a file called metadata.json.
    """

    def _get_metadata(self, dirname):
        prefix = dirname_to_prefix(dirname)
        metadata_json = f'{prefix}-metadata.json'
        md_path = osp.join(dirname, metadata_json)

        if not osp.exists(md_path):
            # Try to see if we are so fast that there exists a temp folder 
            # with the metadata we need.
            md_path = osp.join(dirname + '_temp', metadata_json)

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

    def _saver(self, dirname, metadata, **kwargs):
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

        return FileSaver(dirname, metadata=metadata, **kwargs)


@export
class FileSaver(strax.Saver):
    """Saves data to compressed binary files"""
    json_options = dict(sort_keys=True, indent=4)
    # When writing chunks, rewrite the json file every time we write a chunk
    _flush_md_for_every_chunk = True

    def __init__(self, dirname, metadata, **kwargs):
        super().__init__(metadata, **kwargs)
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

    def _chunk_filename(self, chunk_info):
        if 'filename' in chunk_info:
            return chunk_info['filename']
        ichunk = '%06d' % chunk_info['chunk_i']
        return f'{self.prefix}-{ichunk}'

    def _save_chunk(self, data, chunk_info, executor=None):
        filename = self._chunk_filename(chunk_info)

        fn = os.path.join(self.tempdirname, filename)
        kwargs = dict(data=data, compressor=self.md['compressor'])
        if executor is None:
            filesize = strax.save_file(fn, **kwargs)
            return dict(filename=filename, filesize=filesize), None
        else:
            return dict(filename=filename), executor.submit(
                strax.save_file, fn, **kwargs)

    def _save_chunk_metadata(self, chunk_info):
        is_first = chunk_info['chunk_i'] == 0
        if is_first:
            self.md['start'] = chunk_info['start']

        if self.is_forked:
            # Do not write to the main metadata file to avoid race conditions
            # Instead, write a separate metadata.json file for this chunk,
            # to be collected later.

            # We might not have a filename yet:
            # the chunk is not saved when it is empty
            filename = self._chunk_filename(chunk_info)

            fn = f'{self.tempdirname}/metadata_{filename}.json'
            with open(fn, mode='w') as f:
                f.write(json.dumps(chunk_info, **self.json_options))

        # To ensure we have some metadata to load with allow_incomplete,
        # modify the metadata immediately for the first chunk.
        # If we are forked, modifying self.md is harmless since
        # we're in a different process.

        if not self.is_forked or is_first:
            # Just append and flush the metadata
            # (maybe not super-efficient to write the json every time...
            # just don't use thousands of chunks)
            self.md['chunks'].append(chunk_info)
            if self._flush_md_for_every_chunk:
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


@export
def rechunker(source_directory:str,
              dest_directory: typing.Optional[str] = None,
              replace: bool = False,
              compressor: typing.Optional[str] = None,
              target_size_mb: typing.Optional[str] = None,
              rechunk: bool = True,
              )-> dict:
    """
    Rechunk/Recompress a strax-datatype saved in a FileSystemBackend
    outside of a strax.Context. For a user-friendly context centered
    alternative, see strax.Context.copy_to_frontend:
    github.com/AxFoundation/strax/blob/a0d51fdd3bea52c228c8f74c614fc77bb7cf1bc5/strax/context.py#L1792  # noqa

    One can either specify a destination directory where to store a new
    copy of this data with <dest_directory> or replace the input file
    with it's rechunked version.

    This function can either:
     - rechunk (if <rechunk? is True), probably incrementing the
        <target_size_mb> is also useful (create larger chunks)
     - recompress (if a <compressor> is specified)

    One can also rechunk and recompress simultaneously

    :param source_directory: Path to a folder containing a single
        strax.DataType.
    :param dest_directory: Head of a folder whereto write new data. If
        nothing is specified, write to a temporary directory.
    :param replace: Delete the source_directory and replace it with it's
        rechunked version
    :param compressor: Compressor to be used in saving the rechunked
        data
    :param target_size_mb: Target size of chunks (uncompressed). As long
        as a chunk is smaller than this many MB, keep adding new MBs
        until the chunk is at least target_size_mb or we run out of
        chunks.
    :param rechunk: Do we want to rechunk?
    :return: Dictionary with some information on the write/load times
        involved.
    """
    if not os.path.exists(source_directory):
        raise FileNotFoundError(f'No file at {source_directory}')
    if not replace and dest_directory is None:
        raise ValueError(f'Specify a destination path <dest_file> when '
                         f'not replacing the original path')
    backend_key = os.path.basename(os.path.normpath(source_directory))

    if dest_directory is None and replace:
        _temp_dir = tempfile.TemporaryDirectory()
        dest_directory = _temp_dir.name
    else:
        _temp_dir = False


    if os.path.basename(os.path.normpath(dest_directory)) != backend_key:
        # New key should be correct! If there is not an exact match,
        # we want to make sure that we append the backend_key correctly
        print(f'Will write to {dest_directory} and make sub-folder {backend_key}')
        dest_directory = os.path.join(dest_directory, backend_key)
    backend = strax.FileSytemBackend()
    meta_data = backend.get_metadata(source_directory)
    source_compressor = meta_data['compressor']

    if compressor is not None:
        meta_data['compressor'] = compressor
    if target_size_mb is not None:
        meta_data['chunk_target_size_mb'] = target_size_mb

    data_loader = backend.loader(source_directory)

    load_time_seconds = []
    def loader():
        """Wrapped loader for bookkeeping load time"""
        while True:
            try:
                t0 = time.time()
                data = next(data_loader)
                # Update target chunk size for re-chunking
                data.target_size_mb = meta_data['chunk_target_size_mb']
                load_time_seconds.append(time.time()-t0)
            except StopIteration:
                return
            yield data
    print(f'Rechunking {source_directory} to {dest_directory}')
    saver = backend._saver(dest_directory, metadata=meta_data)

    write_time_start = time.time()
    saver.save_from(loader(), rechunk=rechunk)
    load_time = sum(load_time_seconds)
    write_time = time.time() - write_time_start - load_time

    if replace:
        print(f'move {dest_directory} to {source_directory}')
        shutil.rmtree(source_directory)
        shutil.move(dest_directory, source_directory)
    if _temp_dir:
        _temp_dir.cleanup()

    return dict(backend_key=backend_key,
                load_time=load_time,
                write_time=write_time,
                uncompressed_mb= sum([x['nbytes'] for x in meta_data['chunks']]) / 1e6,
                source_compressor=source_compressor,
                dest_compressor=meta_data['compressor'],
                )
