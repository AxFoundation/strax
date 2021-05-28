import os.path as osp
import strax
from strax.storage.files import dirname_to_prefix
export, __all__ = strax.exporter()
import strax
import os
import hashlib
from bson import json_util
import json
import glob


class RucioDir(strax.StorageFrontend):
    """
    Threat the rucio mounting point as any other DataDirectory,
    just correct the naming conventions of the folders and
    sub-folders.
    """
    can_define_runs = False
    provide_run_metadata = False

    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This frontend is naive, neither smart nor flexible
        self.readonly = True
        self.path = path
        self.backends = [strax.storage.rucio.rucio(self.path)]

    def _find(self, key, write,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        if allow_incomplete or write:
            raise RuntimeError(f'Allow incomplete/writing is not allowed for '
                               f'{self.__class.__name} since data might not be '
                               f'continuous')
        rucio_did = key_to_rucio_did(key)
        base_dir = did_to_dirname(rucio_did)
        md_file_name = key_to_rucio_meta(key)
        md_path = rucio_path(self.path, md_file_name, base_dir)
        exists = os.path.exists(md_path)
        bk = self.backend_key(key)

        # Check exact match
        if exists and self._all_chunk_stored(md_path, base_dir, key):
            return bk

        if fuzzy_for or fuzzy_for_options:
            matches_to = self._match_fuzzy(key,
                                           base_dir,
                                           fuzzy_for,
                                           fuzzy_for_options)
            if matches_to:
                return matches_to

        raise strax.DataNotAvailable

    def _all_chunk_stored(self, meta_path: str, base_dir: str, key: strax.DataKey) -> bool:
        """
        Check if all the chunks are stored that are claimed in the
        metadata-file
        """
        md = read_md(meta_path)
        for chunk in md.get('chunks', []):
            ch_name = key_to_rucio_chunk(key, chunk['chunk_i'])
            ch_path = rucio_path(self.path, ch_name, base_dir)
            if not os.path.exists(ch_path):
                return False
        return True

    def _match_fuzzy(self,
                     key: strax.DataKey,
                     base_dir: str,
                     fuzzy_for: tuple,
                     fuzzy_for_options: tuple,
                     ) -> tuple:
        mds = glob.glob(self.path + f'/xnt_{key.run_id}/*/*/{key.data_type}*metadata.json')
        for md in mds:
            md_dict = read_md(md)
            if self._matches(md_dict['lineage'],
                             key.lineage,
                             fuzzy_for,
                             fuzzy_for_options):
                fuzzy_lineage_hash = md_dict['lineage_hash']
                dirname = f'{key.run_id}-{key.data_type}-{fuzzy_lineage_hash}'
                fuzzy_key = strax.DataKey(run_id=key.run_id,
                                          data_type=key.data_type,
                                          lineage=md_dict['lineage'])
                self.log.warning(f'Was asked for {key} returning {md}')
                if self._all_chunk_stored(md, base_dir, fuzzy_key):
                    return self.backends[0].__class__.__name__, dirname

    def backend_key(self, datakey: strax.DataKey) -> tuple:
        did = key_to_rucio_did(datakey)
        return self.backends[0].__class__.__name__, did_to_dirname(did)

@export
class rucio(strax.StorageBackend):
    """Get data from a rucio directory
    """
    def __init__(self, root_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir

    def get_metadata(self, dirname: str, **kwargs):
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


def key_to_rucio_did(key: strax.DataKey) -> str:
    """Convert a strax.datakey to a rucio did field in rundoc"""
    return f'xnt_{key.run_id}:{key.data_type}-{key.lineage_hash}'


def parse_did(did: str) -> tuple:
    """Parses a Rucio DID and returns a tuple of (number:int, dtype:str, hash: str)"""
    scope, name = did.split(':')
    number = int(scope.split('_')[1])
    dtype, hsh = name.split('-')
    return number, dtype, hsh


def did_to_dirname(did):
    """Takes a Rucio dataset DID and returns a dirname like used by strax.FileSystemBackend"""
    # make sure it's a DATASET did, not e.g. a FILE
    if len(did.split('-')) != 2:
        raise RuntimeError(f"The DID {did} does not seem to be a dataset DID. Is it possible you passed a file DID?")
    dirname = did.replace(':', '-').replace('xnt_', '')
    return dirname


def rucio_path(root_dir: str, filename: str, dirname: str) -> str:
    """Convert target to path according to rucio convention"""
    scope = "xnt_"+dirname.split('-')[0]
    rucio_did = "{0}:{1}".format(scope, filename)
    rucio_md5 = hashlib.md5(rucio_did.encode('utf-8')).hexdigest()
    t1 = rucio_md5[0:2]
    t2 = rucio_md5[2:4]
    return os.path.join(root_dir, scope, t1, t2, filename)


def key_to_rucio_key(key: strax.DataKey) -> str:
    return f'{key.run_id}:{key.data_type}-{key.lineage_hash}'


def key_to_rucio_meta(key: strax.DataKey) -> str:
    return f'{str(key.data_type)}-{key.lineage_hash}-metadata.json'


def key_to_rucio_chunk(key: strax.DataKey, chunk_i: int) -> str:
    return f'{str(key.data_type)}-{key.lineage_hash}-{chunk_i:06}'


def read_md(path: str) -> json:
    with open(path, mode='r') as f:
        md = json.loads(f.read(),
                        object_hook=json_util.object_hook)
    return md
