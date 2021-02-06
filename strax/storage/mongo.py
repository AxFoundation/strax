"""I/O format for MongoDB

This plugin is designed with data monitoring in mind, to put smaller
amounts of extracted data into a database for quick access. However
it should work with any plugin.

Note that there is no check to make sure the 16MB document size
limit is respected!
"""

import strax
import numpy as np
from pymongo import MongoClient, DESCENDING
from strax import StorageFrontend, StorageBackend, Saver
from datetime import datetime
from pytz import utc as py_utc
from warnings import warn
from sys import getsizeof
export, __all__ = strax.exporter()

# Some data is stored in the buffer. Delete when either of these values
# are exceeded
DEFAULT_MONGO_BACKEND_BUFFER_MB = 200
DEFAULT_MONGO_BACKEND_BUFFER_NRUNS = 5


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

        # Attributes for the chunks-buffer
        self.chunks_registry = {}
        self._buffered_backend_keys = []
        self._buff_mb = DEFAULT_MONGO_BACKEND_BUFFER_MB
        self._buff_nruns = DEFAULT_MONGO_BACKEND_BUFFER_NRUNS

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        """See strax.Backend"""
        chunk_i = chunk_info["chunk_i"]
        registry_key = backend_key + str(chunk_i)

        # Build the chunk-registry if not done already, also rebuild if
        # the key is not in the registry (will fail below if also not
        # there on rebuild).
        if registry_key not in self.chunks_registry.keys():
            self._build_chunk_registry(backend_key)

        # Unpack info about this chunk from the query. Return empty if
        # not available. Use a *string* in the registry to lookup the
        # chunk-data (like we do in _build_chunk_registry).
        doc = self.chunks_registry.get(registry_key, None)

        if doc is None:
            # Did not find the data. NB: can be that the query is off in
            # the _build_chunk_registry. In case you end up here but did
            # not expect that, double check that self.chunks_registry is
            # not an empty dict!
            raise ValueError(
                f'Metadata claims chunk{chunk_i} exists but it is unknown to '
                f'the chunks_registry')
        else:
            chunk_doc = doc.get('data', None)
            if chunk_doc is None:
                raise ValueError(
                    f'Doc for chunk_{chunk_i} in wrong format:\n{doc}')

        # Convert JSON to numpy
        chunk_len = len(chunk_doc)
        result = np.zeros(chunk_len, dtype=dtype)
        for i in range(chunk_len):
            for key in np.dtype(dtype).names:
                result[i][key] = chunk_doc[i][key]
        return result

    def _saver(self, key, metadata, **kwargs):
        """See strax.Backend"""
        # Use the key to make a collection otherwise, use the backend-key
        col = self.db[self.col_name if self.col_name is not None else str(key)]
        return MongoSaver(key, metadata, col, **kwargs)

    def get_metadata(self, key):
        """See strax.Backend"""
        query = backend_key_to_query(key)

        # Make sure to get the last of the meta-data docs. Otherwise we
        # might be getting a previously failed document. Sort argument
        # should be obsolete (due to the self.col.delete_many in the
        # MongoSaver) but rather safe than sorry.
        doc = self.db[self.col_name].find_one({
            **query, 'provides_meta': True},
            sort=[('write_time', DESCENDING)])
        if doc and 'metadata' in doc:
            return doc['metadata']
        raise strax.DataNotAvailable

    def _build_chunk_registry(self, backend_key):
        """
        Build chunk info in a single registry using only one query to
        the database. This is much faster as one does not have to do
        n-chunk queries to the database. Just one will do. As the
        documents-size is limited to 16 MB, it's unlikely that we will
        run into memory issues (that we otherwise would not run into).

        :param backend_key: strax.DataKey to query the collection for
        """

        query = backend_key_to_query(backend_key)
        chunks_registry = self.db[self.col_name].find(
            {**query, 'provides_meta': False},
            {"chunk_i": 1, "data": 1})

        # We are going to convert this to a dictionary as that is
        # easier to lookup
        for doc in chunks_registry:
            chunk_key = doc.get('chunk_i', None)
            if chunk_key is None:
                # Should not happen because of the projection in find
                # but let's double check:
                raise ValueError(
                    f'Projection failed, got doc with no "chunk_i":\n{doc}')
            # Update our registry with this chunks info. Use chunk_i as
            # chunk_key. Make it a *string* to avoid potential key-error
            # issues or json-encoding headaches.
            self.chunks_registry[backend_key + str(chunk_key)] = doc.copy()

        # Some bookkeeping to make sure we don't buffer too much in this
        # backend. We still need to return at least one hence the 'and'.
        # See: https://github.com/AxFoundation/strax/issues/346
        if backend_key not in self._buffered_backend_keys:
            self._buffered_backend_keys.append(backend_key)
        while ((getsizeof(self.chunks_registry) / 1e6 > self._buff_mb
                and len(self._buffered_backend_keys) > 1)
               or len(self._buffered_backend_keys) > self._buff_nruns):
            self._clean_first_key_from_registry()

    def _clean_first_key_from_registry(self):
        """
        Remove the first item in the self.buffered_keys and all the
        associated keys in the self.chunks_registry to limit RAM-usage
        """
        # only clean the first entry from the list
        to_clean = self._buffered_backend_keys[0]
        for registry_key in list(self.chunks_registry.keys()):
            if to_clean in registry_key:
                del self.chunks_registry[registry_key]
        del self._buffered_backend_keys[0]


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

    def __init__(self, key, metadata, col, **kwargs):
        """
        Mongo saver
        :param key: strax.Datakey
        :param metadata: metadata to save belonging to data
        :param col: collection (NB! pymongo collection object) of mongo
        instance to write to
        """
        super().__init__(metadata, **kwargs)
        self.col = col
        # All meta_documents should have the key to query against
        basic_meta = backend_key_to_query(key).copy()
        # Start with a clean sheet, we are just going to overwrite
        self.col.delete_many(basic_meta)
        # Add datetime objects as candidates for TTL collections. Either
        # can be used according to the preference of the user to index.
        # Two entries can be used:
        #  1. The time of writing.
        #  2. The time of data taking.
        basic_meta['write_time'] = datetime.now(py_utc)
        # The run_start_time below is a placeholder and will be updated
        # in the _save_chunk_metadata for the first chunk. Nevertheless
        # we need an object in case there e.g. is no chunk.
        basic_meta['run_start_time'] = datetime.now(py_utc)
        # Add flag to doc that we are providing the metadata
        basic_meta['provides_meta'] = True
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
        chunk_i = chunk_info['chunk_i']

        if getattr(data, 'nbytes') > 10_000_000:
            warn('Inserting documents of size > 10 MB, this is getting '
                 'close to the 16 MB document size in mongo',
                 UserWarning)

        aggregate_data = []
        # Remove the numpy structures and parse the data. The dtype
        # information is saved with the metadata so don't worry
        for row in data:
            ins = {}
            for key in list(data.dtype.names):
                ins[key] = row[key]
            ins = remove_np(ins)
            aggregate_data.append(ins)

        # Get the document to update, if none available start a new one
        # for this chunk
        chunk_id = self.ids_chunk.get(chunk_i, None)

        # We can fail here if the document is too large to be written
        # out to mongo. One could do a try: except
        # pymongo.errors.WriteError: pass, but that potentially leads to
        # abuse of a Mongo instance going unnoticed.
        if chunk_id is not None:
            # In principle this should not end up here as each chunk
            # should be it's own document unless you re-chunk
            self.col.update_one({'_id': chunk_id},
                                {'$push': {f'data': aggregate_data}})
        else:
            # Start a new document, update it with the proper information
            doc = self.basic_md.copy()
            doc['write_time'] = datetime.now(py_utc)
            doc['chunk_i'] = chunk_i
            doc["data"] = aggregate_data
            doc['provides_meta'] = False

            chunk_id = self.col.insert_one(doc).inserted_id
            self.ids_chunk[chunk_i] = chunk_id

        return dict(), None

    def _save_chunk_metadata(self, chunk_info):
        """see strax.Saver"""
        # For the first chunk we get the run_start_time and update the
        # run-metadata file
        if int(chunk_info['chunk_i']) == 0:
            self.run_start = datetime.fromtimestamp(
                chunk_info['start']/1e9).replace(tzinfo=py_utc)

        self.col.update_one({'_id': self.id_md},
                            {'$addToSet': {'metadata.chunks': chunk_info}})

    def _close(self):
        """see strax.Saver"""
        # First update the run-starts of all of the chunk-documents as
        # this is a TTL index-candidate
        if self.run_start is not None:
            update = {'run_start_time': self.run_start}
            query = {k: v for k, v in self.basic_md.items()
                     if k in ('number', 'data_type', 'lineage_hash')}
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
    elif isinstance(dictin, (np.ndarray, list)):
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
