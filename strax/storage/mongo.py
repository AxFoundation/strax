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
from datetime import datetime
from pytz import utc as py_utc
export, __all__ = strax.exporter()


@export
class MongoBackend(StorageBackend):
    """Mongo storage backend"""
    def __init__(self, uri, database, col_name=None):
        """
        Backend for reading/writing data from Mongo
        :param uri: Mongo url (with pw and username)
        :param database: name of database (str)
        :param col_name: collection name (str) to look for data
        """

        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.col_name = col_name

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        """See strax.Backend"""
        query = backend_key_to_query(backend_key)
        chunk_name = f'chunk_{chunk_info["chunk_i"]}'

        # Query for the chunk and project the chunk info
        doc = self.db[self.col_name].find_one(
            {**query, "chunk_i": chunk_name},
            {f"data": 1})

        # Unpack info about this chunk from the query. Return empty if not available.
        if doc is None:
            # Did not find the data
            return np.array([], dtype=dtype)
        else:
            chunk_doc = doc.get('data', None)
            if chunk_doc is None:
                raise ValueError(f'Doc for {chunk_name} in wrong format:\n{doc}')

        # Convert JSON to numpy
        chunk_len = len(chunk_doc)
        result = np.zeros(chunk_len, dtype=dtype)
        for i in range(chunk_len):
            for key in np.dtype(dtype).names:
                result[i][key] = chunk_doc[i][key]
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

    def __init__(self, uri, database, col_name=None, *args, **kwargs):
        """
        MongoFrontend for reading/writing data from Mongo
        :param uri: Mongo url (with pw and username)
        :param database: name of database (str)
        :param col_name: collection name (str) to look for data
        :param args: init for StorageFrontend
        :param kwargs: init for StorageFrontend
        """

        super().__init__(*args, **kwargs)
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.backends = [MongoBackend(uri, database, col_name=col_name)]
        self.col_name = col_name

    def _find(self, key, write, allow_incomplete, fuzzy_for,
              fuzzy_for_options):
        """See strax.Frontend"""
        if write:
            return self.backends[0].__class__.__name__, str(key)
        query = backend_key_to_query(str(key))
        if self.db[self.col_name].count_documents(query):
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
        # Add datetime objects as candidates for TTL collections. Either can
        # be used according to the preference of the user to index.
        # Two entries can be used:
        #  1. The time of writing.
        #  2. The time of data taking.
        basic_meta['write_time'] = datetime.now(py_utc)
        # The run_start_time below is a placeholder and will be updated in the
        # _save_chunk_metadata for the first chunk. Nevertheless we need an
        # object in case there e.g. is no chunk.
        basic_meta['run_start_time'] = datetime.now(py_utc)
        # If available later update with this value:
        self.run_start = None
        # This info should be added to all of the associated documents
        self.basic_md = basic_meta

        # For the metadata copy this too:
        meta_data = basic_meta.copy()
        meta_data['metadata'] = self.md

        # Save object_ids for fast querying and updates
        self.id_md = self.col.insert_one(meta_data).inserted_id
        # Also save all the chunks
        self.ids_chunk = {}

    def _save_chunk(self, data, chunk_info, executor=None):
        """see strax.Saver"""
        chunk_number = chunk_info['chunk_i']
        chunk_name = f'chunk_{chunk_number}'

        aggregate_data = []
        # Remove the numpy structures and parse the data. The dtype information
        # is saved with the metadata so don't worry
        for row in data:
            ins = {}
            for key in list(data.dtype.names):
                ins[key] = row[key]
            ins = remove_np(ins)
            aggregate_data.append(ins)

        # Get the document to update, if none available start a new one for
        # this chunk
        chunk_id = self.ids_chunk.get(chunk_name, None)
        if chunk_id is not None:
            self.col.update_one({'_id': chunk_id},
                                {'$addToSet': {f'data': aggregate_data}})
        else:
            # Start a new document, update it with the proper information
            doc = self.basic_md.copy()
            doc['write_time'] = datetime.now(py_utc)
            doc['chunk_i'] = chunk_name
            doc["data"] = aggregate_data

            chunk_id = self.col.insert_one(doc).inserted_id
            self.ids_chunk[chunk_name] = chunk_id

        return dict(), None

    def _save_chunk_metadata(self, chunk_info):
        """see strax.Saver"""
        # For the first chunk we get the run_start_time and update the run-metadata file
        if int(chunk_info['chunk_i']) == 0:
            self.run_start = t0 = datetime.fromtimestamp(
                chunk_info['start']/1e9).replace(tzinfo=py_utc)
            self.col.update_one({'_id': self.id_md},
                                {'$set':
                                     {'run_start_time': t0}})

        self.col.update_one({'_id': self.id_md},
                            {'$addToSet':
                                 {'metadata.chunks': chunk_info}})

    def _close(self):
        """see strax.Saver"""
        # First update the run-starts of all of the chunk-documents as this is
        # a TTL index -candidate
        if self.run_start is not None:
            update = {'run_start_time': self.run_start}
            query = {k: v for k, v in self.basic_md.items()
                     if k in ('run_id', 'data_type', 'lineage_hash')}
            self.col.update_many(query, {'$set': update})

        # Update the metadata
        update = {f'metadata.{k}': v
                  for k, v in self.md.items()
                  if k in ('writing_ended', 'exception')}
        # Also update all of the chunk-documents with the run_start_time
        self.col.update_one({'_id': self.id_md}, {'$set': update})


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
    elif isinstance(dictin, np.floating):
        return float(dictin)
    else:
        return dictin
    return result
