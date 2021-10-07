import unittest
import strax
from strax.testutils import Records, Peaks
import os
import shutil
import tempfile
import pymongo
from warnings import warn


class TestMongoFrontend(unittest.TestCase):
    """
    Test the saving behavior of the context with the strax.MongoFrontend

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just some free ATLAS mongo server, can easily replaced
    with another (free) server.
    """
    run_test = True

    def setUp(self):
        self.test_run_id = '0'
        self.all_targets = ('peaks', 'records')
        self.mongo_target = 'peaks'

        # Just some free ATLAS mongo server, see test-class docstring
        if 'TEST_MONGO_URI' not in os.environ:
            self.run_test = False
            warn('Cannot connect to test-database')
            return

        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test_mongosf_database'
        self.collection_name = 'temp-test-collection-mongosf'
        client = pymongo.MongoClient(uri)
        self.database = client[db_name]
        assert self.collection_name not in self.database.list_collection_names()

        self.mongo_sf = strax.MongoFrontend(uri=uri,
                                            database=db_name,
                                            take_only=self.mongo_target,
                                            col_name=self.collection_name,
                                            )

        self.path = os.path.join(tempfile.gettempdir(), 'strax_data')
        self.local_sf = strax.DataDirectory(self.path)
        self.st = strax.Context(register=[Records, Peaks],
                                storage=[self.local_sf,
                                         self.mongo_sf],
                                use_per_run_defaults=True,
                                )
        assert not self.all_targets_stored

    @property
    def all_targets_stored(self) -> bool:
        return all([self.st.is_stored(self.test_run_id, t) for t in self.all_targets])

    @property
    def stored_in_mongo(self) -> bool:
        return self.st._is_stored_in_sf(self.test_run_id, self.mongo_target, self.mongo_sf)

    @property
    def stored_locally(self) -> bool:
        return all([self.st._is_stored_in_sf(self.test_run_id, t, self.local_sf)
                    for t in self.all_targets])

    def tearDown(self):
        if not self.run_test:
            return

        self.database[self.collection_name].drop()

        if os.path.exists(self.path):
            print(f'rm {self.path}')
            shutil.rmtree(self.path)

    def test_save_and_load(self):
        if not self.run_test:
            return

        # Shouldn't be any traces of data
        assert self.database[self.collection_name].find_one() is None
        assert not self.stored_in_mongo
        assert not self.stored_locally, self.path
        assert not self.all_targets_stored

        # Make ALL the data and check it's stored everywhere
        for t in self.all_targets:
            self.st.make(self.test_run_id, t)
        assert self.database[self.collection_name].find_one() is not None
        # NB: the context writes to ALL the storage frontends that are susceptible
        assert self.stored_in_mongo
        assert self.stored_locally

        # Double check that we can load if we only have the mongo sf and cannot make data
        self.st.storage = [self.mongo_sf]
        self.st.context_config['forbid_creation_of'] = self.all_targets
        assert not self.all_targets_stored
        assert self.stored_in_mongo
        peaks = self.st.get_array(self.test_run_id, self.mongo_target)
        assert len(peaks)

        # Check that lineage changes result in non-loadable data
        self.st._plugin_class_registry['peaks'].__version__ = 'some other version'
        assert not self.stored_in_mongo
        with self.assertRaises(strax.DataNotAvailable):
            self.st.get_array(self.test_run_id, self.mongo_target)

        # For completeness, also check the buffer cleaning works
        mongo_backend = self.mongo_sf.backends[0]
        len_before = len(mongo_backend.chunks_registry)
        mongo_backend._clean_first_key_from_registry()
        assert len(mongo_backend.chunks_registry) < len_before
