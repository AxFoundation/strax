"""I/O format for MongoDB

This plugin is designed for small amounts of basic information and 
is specifically implemented for an online data monitoring system.
Each chunk is processed separately and extracted information is stored
in a single MongoDB document of the following form:

doc = {
  number: run_id,
  plugin: plugin_name,
  chunk: chunk_id,
  data: {  PLUGIN_DATA },
  }
}

Note that you must define Mongo connectivity parameters are environment
variables. Note also that there is a 16 MB size limit on MongoDB documents,
so plugins must be relatively light on data.
"""

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

import strax
import numpy as np
from strax import StorageFrontend, StorageBackend, DataKey
from strax.utils import unpack_dtype
from bson import json_util
import copy

export, __all__ = strax.exporter()

@export
class MongoBackend(StorageBackend):

    def __init__(self, uri, database):
        if MongoClient is None:
            raise ImportError("Pymongo did not import: "
                              "cannot use MongoDBStore")
        self.client = MongoClient(uri)
        self.db = self.client[database]
        
    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):

        doc = self.db[backend_key].find_one(
            {'chunk': chunk_info['chunk_i']})
        if doc is None:
            return []

        result = np.zeros(len(doc['data']), dtype=dtype)
        for i, d in enumerate(doc['data']):
            for key in np.dtype(dtype).names:
                result[key][i] = d[key]
        return result
            
    def _saver(self, key, metadata):
        return MongoSaver(key, metadata, self.db[str(key)])

    def get_metadata(self, key):
        return self.db[key].find_one({'metadata': True})

    
@export
class MongoFrontend(StorageFrontend):
    """MongoDB storage frontend. For high-level data"""

    def __init__(self, uri, database, *args, **kwargs):        
        super().__init__(*args, **kwargs)        
        if MongoClient is None:
            raise ImportError("Pymongo did not import: "
                              "cannot use MongoDBStore")
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.backends = [MongoBackend(uri, database)]
        
    def _find(self, key, write, allow_incomplete, fuzzy_for,
              fuzzy_for_options):

        if write:
            return self.backends[0].__class__.__name__, str(key)
        if self.db[str(key)].count():
            self.log.debug(f"{key} is in cache.")
            return self.backends[0].__class__.__name__, str(key)
        self.log.debug(f"{key} is NOT in cache.")
        raise strax.DataNotAvailable

    def _read_meta(self, coll_name):
        doc = self.db[coll_name].find_one({'metadata': True})
        if doc is not None and 'dtype' in doc:
            doc['dtype'] = np.dtype(doc['dtype'])
        return doc



@export
class MongoSaver(strax.Saver):
    prefer_rechunk = False
    
    def __init__(self, key, metadata, col):
        super().__init__(metadata)
        self.col = col
        self.col.insert(dict(metadata=True, **self.md))
        
    def _save_chunk(self, data, chunk_info, executor):        
        chunk = chunk_info['chunk_i']
        for row in data:
            ins = {}

            for key in list(data.dtype.names):                
                ins[key] = row[key]
            ins = remove_np(ins)
            self.col.update({"chunk": chunk},
                            {"$push": {"data": ins}},
                            upsert=True)
        return dict(), None
                
    def _save_chunk_metadata(self, chunk_info):
        self.col.find_one_and_update(dict(metadata=True),
                                     {'$push': {'chunks': chunk_info}})
        
    def _close(self):
        update = {k: v
                for k, v in self.md.items()
                  if k in ('writing_ended', 'exception')}
        self.col.find_one_and_update(dict(metadata=True),
                                     {'$set': update})

def remove_np(dictin):

    if isinstance(dictin, dict):
        result = {}
        for k in dictin.keys():
            result[k] = remove_np(dictin[k])
    elif isinstance(dictin, np.ndarray) or isinstance(dictin, list):
        result = []
        for k in dictin:
            result.append(remove_np(k))
    elif isinstance(dictin, np.integer):
        return int(dictin)
    elif isinstance(dictin, np.float):
        return float(dictin)
    else:
        return dictin
    return result
