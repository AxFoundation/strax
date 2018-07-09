import os
import os.path as osp
import json
import zipfile

import strax
export, __all__ = strax.exporter()

from .files import run_metadata_filename



@export
class DataZipDirectory(strax.StorageFrontend):
    """ZipFile-based storage frontend for strax.

    All data for one run is assumed to be in a single zip file <run_id>.zip,
    with the same file/directory structure as created by FileStore.

    We cannot write zip files directly (this would result in concurrency hell),
    instead these zip files are made by zipping stuff from FileSytemBackend.
    """

    def __init__(self, path='.', *args, readonly=True, **kwargs):
        if not readonly:
            raise NotImplementedError("Zipfiles are currently read-only")
        super().__init__(*args, readonly=readonly, **kwargs)
        self.path = path
        if not osp.exists(path):
            os.makedirs(path)

    def _find(self, key,
              write, ignore_versions, ignore_config):
        assert not write

        # Check exact match / write case
        bk = self._backend_key(key)
        with zipfile.ZipFile(self._zipname(key)) as zp:
            try:
                zp.getinfo(str(key) + '/metadata.json')
                return bk
            except KeyError:
                pass

            if not ignore_versions and not ignore_config:
                raise strax.DataNotAvailable

        raise NotImplementedError("Fuzzy matching within zipfiles not yet "
                                  "implemented")

    def run_metadata(self, run_id):
        with zipfile.ZipFile(self._zipname(run_id)) as zp:
            try:
                with zp.open(run_metadata_filename % run_id) as f:
                    return json.loads(f.read())
            except KeyError:
                raise strax.RunMetadataNotAvailable

    def write_run_metadata(self, run_id, metadata):
        raise NotImplementedError("Zipfiles cannot write")

    def remove(self, key):
        raise NotImplementedError("Zipfiles cannot write")

    def _set_write_complete(self, key):
        raise NotImplementedError("Zipfiles cannot write")

    def _backend_key(self, key):
        return self._zipname(key), str(key)

    def _zipname(self, key):
        zipname = osp.join(self.path, key.run_id)
        # Since we're never writing, this check can be here
        if not osp.exists(zipname):
            raise strax.DataNotAvailable


class ZipFileBackend(strax.StorageBackend):

    def _read_chunk(self, zipn_and_dirn, chunk_info, dtype, compressor):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            with zp.open(dirn + '/' + chunk_info['filename']) as f:
                return strax.load_file(f, dtype=dtype, compressor=compressor)

    def get_metadata(self, zipn_and_dirn):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            with zp.open(dirn + '/metadata.json') as f:
                return json.loads(f.read())

    def saver(self, *args, **kwargs):
        raise NotImplementedError("Zipfiles cannot write")
