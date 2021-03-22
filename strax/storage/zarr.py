import glob
import json
import tempfile
import os
import time
import os.path as osp
from ast import literal_eval
from bson import json_util
import shutil
import zarr
from copy import deepcopy
import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()


@export
class Zarr(StorageFrontend):
    """
    Zarr storage interface. 
    This interface is mostly for testing zarr and whether it
    should replace the entire strax storage system.
    Usage:
    zstorage = strax.Zarr("./data/strax.zarr", deep_scan=True, overwrite="always")
    ctx.storage.append(zstorage)
    """

    can_define_runs = True
    provide_run_metadata = False

    def __init__(self, *args, store='./strax_data.zarr', 
                    deep_scan=False, seperator="/", **kwargs):
        """
        :param path: Path to folder with data subfolders.
        :param deep_scan: Let scan_runs scan over folders,
        so even data for which no run-level metadata is available
        is reported.

        For other arguments, see DataRegistry base class.
        """
        super().__init__(*args, **kwargs)
        if isinstance(store, str):
            store = zarr.DirectoryStore(store)
        # self.store = store
        self.backends = [strax.ZarrBackend(store, **kwargs)]
        self.deep_scan = deep_scan
        self.sep = seperator
        
    def run_metadata(self, run_id, projection=None):
        for sb in self.backends:
            try:
                md = sb.get_metadata(run_id)
                break
            except strax.DataNotAvailable:
                pass
        else:
            raise strax.RunMetadataNotAvailable(
                f"Cannot find run metadata for {run_id}")
        
        md = strax.flatten_run_metadata(md)
        if projection is not None:
            md = {k: v
                  for k, v in md.items()
                  if k in projection}
        return md

    def write_run_metadata(self, run_id, metadata):
        for sb in self.backends:
            try:
                sb.create_group(run_id)
                sb.write_metadata(run_id, metadata)
                break
            except:
                pass
        else:
            raise strax.DataNotAvailable(
                f"Cannot write run metadata for {run_id}")

    def _scan_runs(self, store_fields):
        """Iterable of run document dictionaries.
        These should be directly convertable to a pandas DataFrame.
        """
        for sb in self.backends:
            for run_id in sb.keys():
                try:
                    md = sb.get_metadata(run_id)
                    yield md
                except:
                    if self.deep_scan:
                        yield dict(name=run_id)
                
    def _find(self, key, write,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        
        bk = backend, path = self.backend_key(key)
        exists = self._get_backend(backend).exists(path)
        if write:
            if exists and not self._can_overwrite(key):
                raise strax.DataExistsError(at=bk)
            return bk
        elif exists:
            return bk
        raise strax.DataNotAvailable

    def _parse_folder_name(self, fn):
        """Return (run_id, data_type, hash) if folder name matches
        DataDirectory convention, raise InvalidFolderNameFormat otherwise
        """
        stuff = fn.split(self.sep)
        if len(stuff) < 3:
            # This is not a folder with strax data
            raise InvalidFolderNameFormat(fn)
            stuff = stuff[-3:]
        return stuff

    def backend_key(self, key):
        path = self.sep.join([key.run_id, key.data_type, key.lineage_hash])
        return self.backends[0].__class__.__name__, path

    def remove(self, key):
        backend, path = self.backend_key(key)
        self._get_backend(backend).remove(path)


@export
class ZarrBackend(strax.StorageBackend):
    """Store data in a zarr.

    """

    def __init__(self, store, **kwargs):
        self.root = zarr.group(store=store, **kwargs)

    def keys(self):
        yield from list(self.root.keys())
    
    def exists(self, path):
        return self.root.get(path, None) is not None

    def get_metadata(self, path):
        if "strax_metadata" in self.root[path].attrs:
            md = self.root[path].attrs["strax_metadata"]
            return deepcopy(md)
        else:
            raise strax.DataNotAvailable(f"No metadata found at path {path}")

    def write_metadata(self, path, metadata):
        try:
            self.root[path].attrs["strax_metadata"] = metadata
        except:
            raise strax.DataNotAvailable(f"No metadata found at path {path}")

    def create_group(self, path):
        if path not in self.root:
            self.root.create_group(path)

    def _read_chunk(self, path, chunk_info, dtype, compressor):
        z = self.root.get(path, None)
        if z is None:
            raise strax.DataNotAvailable(f"No data found at path {path}")
        md = deepcopy(z.attrs["strax_metadata"])
        chunks = md["chunks"]
        chunk_len = chunk_info["n"]
        start = sum([chunk["n"] for chunk in chunks if chunk["chunk_i"]<chunk_info["chunk_i"]])
        array = z[start:start+chunk_len]
        array.dtype = dtype
        return array
    
    def _saver(self, path, metadata, **kwargs):
        if path not in self.root:
            dtype = [(type_[0][1],type_[1]) for type_ in literal_eval(metadata["dtype"])]
            array = self.root.empty(path, dtype=dtype, shape=(0,))
        else:
            array = self.root[path]
        
        return strax.ZarrSaver(array, metadata, **kwargs)

    def remove(self, path):
        self.root.pop(path, None)
        

@export
class ZarrSaver(strax.Saver):
    """Saves data to a zarr group"""
    
    def __init__(self, array, metadata, **kwargs):
        super().__init__(metadata, **kwargs)
        self.array = array
        self._flush_metadata()

    def _save_chunk(self, data, chunk_info, executor=None):
        if executor is not None:
            future = executor.submit(self.array.append, data)
        else:
            self.array.append(data)
            future = None
        # self._save_chunk_metadata(chunk_info)
        return {}, future

    def _flush_metadata(self):
        self.array.attrs["strax_metadata"] = self.md

    def _save_chunk_metadata(self, chunk_info):
        if chunk_info['chunk_i'] == 0:
            self._start = chunk_info['start']
        self.md['start'] = chunk_info['start']

        self.md['chunks'].append(chunk_info)
        self._flush_metadata()
        # self._flush_metadata()

    def _close(self):
        self._flush_metadata()

