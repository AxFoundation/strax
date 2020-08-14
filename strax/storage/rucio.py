import json
import hashlib
import os.path as osp

import strax
from strax.storage.files import dirname_to_prefix

export, __all__ = strax.exporter()


@export
class rucio(strax.StorageBackend):
    """Get data from a rucio directory
    """
    def __init__(self, root_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir

    def get_metadata(self, dirname:str, **kwargs):
        prefix = dirname_to_prefix(dirname)
        metadata_json = f'{prefix}-metadata.json'
        fn = rucio_path(self.root_dir, metadata_json, dirname)
        folder = osp.join('/', *fn.split('/')[:-1])
        if not osp.exists(folder):
            raise strax.DataNotAvailable(f"No folder for matadata at {fn}")
        if not osp.exists(fn):
            raise strax.DataCorrupted(f"Folder exists but no matadata at {fn}")

        with open(fn, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        fn = rucio_path(self.root_dir, chunk_info['filename'], dirname)
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata):
        raise NotImplementedError(
            "Cannot save directly into rucio, upload with admix instead")


def rucio_path(root_dir, filename, dirname):
    """Convert target to path according to rucio convention"""
    scope = "xnt_"+dirname.split('-')[0]
    rucio_did = "{0}:{1}".format(scope, filename)
    rucio_md5 = hashlib.md5(rucio_did.encode('utf-8')).hexdigest()
    t1 = rucio_md5[0:2]
    t2 = rucio_md5[2:4]
    return osp.join(root_dir, scope, t1, t2, filename)
