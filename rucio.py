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

    def get_metadata(self, dirname, **kwargs):
        dirname = str(dirname)
        prefix = dirname_to_prefix(dirname)
        metadata_json = f'{prefix}-metadata.json'
        fn = rucio_path(metadata_json, dirname)
        with open(fn, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        #print('yes')
        fn = rucio_path(chunk_info['filename'], dirname)
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata):
        raise NotImplementedError(
            "Cannot save directly into rucio, upload with admix instead")


def rucio_path(filename, dirname):
    root_path ='/dali/lgrandi/rucio'
    scope = "xnt_"+dirname.split('-')[0]
    rucio_did = "{0}:{1}".format(scope,filename)
    rucio_md5 = hashlib.md5(rucio_did.encode('utf-8')).hexdigest()
    t1 = rucio_md5[0:2]
    t2 = rucio_md5[2:4]
    return osp.join(root_path,scope,t1,t2,filename)
