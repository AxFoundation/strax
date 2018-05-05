import numpy as np

import strax
from .common import Store, Saver, NotCached

export, __all__ = strax.exporter()

try:
    import pymongo
except ImportError:
    pymongo = None


@export
class MongoStore(Store):
    """MongoDB-based storage backend for strax, intended
    for high-level data.

    Data is stored in uncompressed / fully expanded documents (not binary
    blobs).
    A _chunk_i: integer field is added to each document.

    Metadata is stored in the same collection as data, but with a
    metadata = True attribute in the document.
    """

    def __init__(self, uri, database='strax_data', *args, **kwargs):
        """
        :param connection_uri: Connection URI, which also specifies DB
        {provides_doc}
        """
        super().__init__(*args, **kwargs)
        if pymongo is None:
            raise ImportError("Pymongo did not import: "
                              "cannot use MongoDBStore")

        self.client = pymongo.MongoClient(uri)
        self.client.admin.command('ping')
        self.db = self.client[database]

    def _find(self, key):
        if self.db[str(key)].count():
            self.log.debug(f"{key} is in cache.")
            return str(key)

        self.log.debug(f"{key} is NOT in cache.")
        raise NotCached

    def _read_chunk(self, coll_name, chunk_info, dtype, compressor):
        docs = list(self.db[coll_name].find(
            {'_chunk_i': chunk_info['chunk_i']}))

        result = np.zeros(len(docs), dtype=dtype)
        for i, d in enumerate(docs):
            for k, v in d.items():
                if k in ('_chunk_i', '_id'):
                    continue
                result[i][k] = v
        return result

    def _read_meta(self, coll_name):
        return self.db[coll_name].find_one({'metadata': True})

    def saver(self, key, metadata):
        super().saver(key, metadata)
        return MongoSaver(key, metadata, self.db[str(key)])


@export
class MongoSaver(Saver):
    prefer_rechunk = False

    def __init__(self, key, metadata, col):
        super().__init__(key, metadata)
        self.col = col
        self.col.insert(dict(metadata=True, **self.md))

    def _save_chunk(self, data, chunk_info):
        docs = [dict(_chunk_i=chunk_info['chunk_i'])
                for _ in range(len(data))]
        for i, d in enumerate(docs):
            for k in data.dtype.names:
                docs[i][k] = np.asscalar(data[i][k])

        self.col.insert(docs)
        return dict()

    def _save_chunk_metadata(self, chunk_info):
        self.col.find_one_and_update(dict(metadata=True),
                                     {'$push': {'chunks': chunk_info}})

    def close(self):
        super().close()
        update = {k: v
                  for k, v in self.md.items()
                  if k in ('writing_ended', 'exception')}
        self.col.find_one_and_update(dict(metadata=True),
                                     {'$set': update})
