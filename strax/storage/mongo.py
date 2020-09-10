"""I/O format for MongoDB

This plugin is designed with data monitoring in mind, to put smaller
amounds of extracted data into a database for quick access. However
it should work with any plugin.

Note that there is no check to make sure the 16MB document size
limit is respected!
"""

import strax
import numpy as np
from pymongo import MongoClient
from strax import StorageFrontend, StorageBackend, Saver

export, __all__ = strax.exporter()


@export
class MongoBackend(StorageBackend):
    """Mongo storage backend"""
    def __init__(self, uri, database, col_n=None):
        """
        Backend for reading/writing data from Mongo
        :param uri: Mongo url (with pw and username)
        :param database: name of database (str)
        :param col_n: collection name (str) to look for data
        """
        if MongoClient is None:
            raise ImportError("Pymongo did not import: "
                              "cannot use MongoDBStore")
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.col_name = col_n

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        """See strax.Backend"""
        query = backend_key_to_query(backend_key)
        chunk_name = f'chunk_{chunk_info["chunk_i"]}'
        # Query for the chunk and project the chunk info
        doc = self.db[self.col_name].find_one(
            {**query, chunk_name: {"$exists": True}},
            {f"{chunk_name}.$": 1})
        doc = doc.get(chunk_name, None)

        if doc is None:
            return np.array([], dtype=dtype)

        # Convert JSON to numpy
        result = np.zeros(len(doc), dtype=dtype)
        for i, d in enumerate(doc):
            for key in np.dtype(dtype).names:
                result[i][key] = d['data'][key]
        return result

    def _saver(self, key, metadata):
        """See strax.Backend"""
        # Use the key to make a collection otherwise, use the backend-key
        col = self.db[self.col_name if self.col_name is not None else str(key)]
        return MongoSaver(key, metadata, col)

    def get_metadata(self, key):
        """See strax.Backend"""
        query = backend_key_to_query(key)
        doc = self.db[self.col_name].find_one(query)
        if doc and 'metadata' in doc:
            return doc['metadata']
        raise strax.DataNotAvailable


@export
class MongoFrontend(StorageFrontend):
    """MongoDB storage frontend"""

    def __init__(self, uri, database, col_n=None, *args, **kwargs):
        """
        MongoFrontend for reading/writing data from Mongo
        :param uri: Mongo url (with pw and username)
        :param database: name of database (str)
        :param col_n: collection name (str) to look for data
        :param args: init for StorageFrontend
        :param kwargs: init for StorageFrontend
        """

        super().__init__(*args, **kwargs)
        if MongoClient is None:
            raise ImportError("Pymongo did not import: "
                              "cannot use MongoDBStore")
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.backends = [MongoBackend(uri, database, col_n=col_n)]
        self.col_name = col_n

    def _find(self, key, write, allow_incomplete, fuzzy_for,
              fuzzy_for_options):
        """See strax.Frontend"""
        if write:
            return self.backends[0].__class__.__name__, str(key)
        query = backend_key_to_query(str(key))
        if self.db[self.col_name].find_one(query):
            self.log.debug(f"{key} is in cache.")
            return self.backends[0].__class__.__name__, str(key)
        self.log.debug(f"{key} is NOT in cache.")
        raise strax.DataNotAvailable


@export
class MongoSaver(Saver):
    allow_rechunk = False

    def __init__(self, key, metadata, col):
        """
        Mongo saver
        :param key: strax.Datakey
        :param metadata: metadata to save belonging to data
        :param col: collection (NB! pymongo collection object) of mongo
        instance to write to
        """
        super().__init__(metadata)
        self.col = col
        # Parse basic properties for online document by forcing keys in specific
        # representations (rep)
        basic_meta = {}
        for k, rep in (
                ('run_id', int), ('data_type', str), ('lineage_hash', str)):
            basic_meta[k.replace('run_id', 'number')] = rep(self.md[k])
        basic_meta['metadata'] = self.md

        # Save object id to write other data to
        self.id = self.col.insert_one(basic_meta).inserted_id

    def _save_chunk(self, data, chunk_info, executor):
        """see strax.Saver"""
        chunk_number = chunk_info['chunk_i']

        # Remove the numpy structures and parse the data. The dtype information
        # is saved with the metadata so don't worry
        for row in data:
            ins = {}
            for key in list(data.dtype.names):
                ins[key] = row[key]
            ins = remove_np(ins)
            self.col.update_one({'_id': self.id},
                                {'$addToSet': {f'chunk_{chunk_number}':
                                                   {"data": ins}}
                                 })
        return dict(), None

    def _save_chunk_metadata(self, chunk_info):
        """see strax.Saver"""
        self.col.update_one({'_id': self.id},
                            {'$addToSet':
                                 {'metadata.chunks': chunk_info}})

    def _close(self):
        """see strax.Saver"""
        update = {f'metadata.{k}': v
                  for k, v in self.md.items()
                  if k in ('writing_ended', 'exception')}
        self.col.update_one({'_id': self.id}, {'$set': update})


def backend_key_to_query(backend_key):
    """Convert backend key to queryable dictionary"""
    n, d, l = backend_key.split('-')
    return {'number': int(n), 'data_type': d, 'lineage_hash': l}


def remove_np(dictin):
    """Remove numpy types from a dict so it can be inserted into
    mongo."""
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
