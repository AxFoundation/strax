import json
import os
import os.path as osp
import shutil
import zipfile

import strax
from .files import RUN_METADATA_PATTERN

export, __all__ = strax.exporter()


@export
class ZipDirectory(strax.StorageFrontend):
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
        self.backends = [ZipFileBackend()]
        self.path = path
        if not osp.exists(path):
            os.makedirs(path)

    def _find(self, key, write,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        assert not write

        # Check exact match / write case
        bk = self._backend_key(key)
        with zipfile.ZipFile(self._zipname(key)) as zp:
            try:
                dirname = str(key)
                prefix = strax.dirname_to_prefix(dirname)
                zp.getinfo(f'{dirname}/{prefix}-metadata.json')
                return bk
            except KeyError:
                pass

            if not len(fuzzy_for) and not len(fuzzy_for_options):
                raise strax.DataNotAvailable

        raise NotImplementedError("Fuzzy matching within zipfiles not yet "
                                  "implemented")

    def run_metadata(self, run_id):
        with zipfile.ZipFile(self._zipname(run_id)) as zp:
            try:
                with zp.open(RUN_METADATA_PATTERN % run_id) as f:
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
        return (self.backends[0].__class__.__name__,
                (self._zipname(key), str(key)))

    def _zipname(self, key):
        zipname = osp.join(self.path, key.run_id + '.zip')
        # Since we're never writing, this check can be here
        # TODO: sounds like a bad idea?
        if not osp.exists(zipname):
            raise strax.DataNotAvailable
        return zipname

    @staticmethod
    def zip_dir(input_dir, output_zipfile, delete=False):
        """Zips subdirectories of input_dir to output_zipfile
        (without compression).
        Travels into subdirectories, but not sub-subdirectories.
        Skips any other files in directory.
        :param delete: If True, delete original directories
        """
        with zipfile.ZipFile(output_zipfile, mode='w') as zp:
            for dirn in os.listdir(input_dir):
                full_dirn = os.path.join(input_dir, dirn)
                if not osp.isdir(full_dirn):
                    continue
                for fn in os.listdir(full_dirn):
                    zp.write(os.path.join(full_dirn, fn),
                             arcname=os.path.join(dirn, fn))
                if delete:
                    shutil.rmtree(full_dirn)


@export
class ZipFileBackend(strax.StorageBackend):

    def _read_chunk(self, zipn_and_dirn, chunk_info, dtype, compressor):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            with zp.open(dirn + '/' + chunk_info['filename']) as f:
                return strax.load_file(f, dtype=dtype, compressor=compressor)

    def get_metadata(self, zipn_and_dirn):
        zipn, dirn = zipn_and_dirn
        with zipfile.ZipFile(zipn) as zp:
            prefix = strax.dirname_to_prefix(dirn)
            with zp.open(f'{dirn}/{prefix}-metadata.json') as f:
                return json.loads(f.read())

    def saver(self, *args, **kwargs):
        raise NotImplementedError("Zipfiles cannot write")
