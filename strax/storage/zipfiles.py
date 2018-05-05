import os
import json
import zipfile

import strax
from .files import FileStore
from .common import NotCached

export, __all__ = strax.exporter()


@export
class ZipFileStore(FileStore):
    """ZipFile-based storage backend for strax.
    All data for one run is assumed to be in a single zip file <run_id>.zip,
    with the same file/directory structure as created by FileStore.

    This cannot modify the zip files (would result in concurrency hell).
    # TODO: make method to do create zip file more conveniently
    """
    def __init__(self, *args, **kwargs):
        """:param data_dirs: List of data directories to use.
        Entries are preferred for use from first to last.

        {provides_doc}
        """
        super().__init__(*args, **kwargs)
        self.readonly = True

    @staticmethod
    def _get_fn(key):
        return key.data_type + '_' + strax.deterministic_hash(key.lineage)

    def _find(self, key):
        dirn = str(key)

        for zipn in self._candidate_zips(key):
            with zipfile.ZipFile(zipn) as zp:
                try:
                    zp.getinfo(dirn + '/metadata.json')
                except KeyError:
                    continue
                else:
                    self.log.debug(f"{key} is in cache.")
                    return zipn, dirn

        self.log.debug(f"{key} is NOT in cache.")
        raise NotCached

    def _candidate_zips(self, key):
        """Return zipfiles at which data meant by key
        could be found or saved"""
        return [
            os.path.join(d, key.run_id + '.zip')
            for d in self.data_dirs]

    def _read_chunk(self, zipn_and_dirn, chunk_info, dtype, compressor):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            with zp.open(dirn + '/' + chunk_info['filename']) as f:
                return strax.load_file(f, dtype=dtype, compressor=compressor)

    def _read_meta(self, zipn_and_dirn):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            with zp.open(dirn + '/metadata.json') as f:
                return json.loads(f.read())

    def saver(self, key, metadata):
        raise NotImplementedError
