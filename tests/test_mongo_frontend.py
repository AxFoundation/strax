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

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.
    """
    run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        if 'TEST_MONGO_URI' not in os.environ:
            self.run_test = False
            warn('Cannot connect to test-database')
            return

        self.test_run_id = '0'
        self.all_targets = ('peaks', 'records')
        self.mongo_target = 'peaks'

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

        self.st = strax.Context(register=[Records, Peaks],
                                storage=[self.mongo_sf],
                                use_per_run_defaults=True,
                                )
        assert not self.is_all_targets_stored

    def tearDown(self):
        if not self.run_test:
            return
        self.database[self.collection_name].drop()

    @property
    def is_all_targets_stored(self) -> bool:
        """This should always be False as one of the targets (records) is not stored in mongo"""
        return all([self.st.is_stored(self.test_run_id, t) for t in self.all_targets])

    @property
    def is_stored_in_mongo(self) -> bool:
        return self.st._is_stored_in_sf(self.test_run_id, self.mongo_target, self.mongo_sf)

    @property
    def is_data_in_collection(self):
        return self.database[self.collection_name].find_one() is None

    def test_write_and_load(self):
        if not self.run_test:
            return

        # Shouldn't be any traces of data
        assert not self.is_data_in_collection
        assert not self.is_stored_in_mongo
        assert not self.is_all_targets_stored

        # Make ALL the data
        # NB: the context writes to ALL the storage frontends that are susceptible
        for t in self.all_targets:
            self.st.make(self.test_run_id, t)

        assert self.is_data_in_collection
        assert self.is_stored_in_mongo
        msg = ("since take_only of the MongoFrontend is specified not all data "
               "should be in stored (records should not be in any frontend)")
        assert not self.is_all_targets_stored, msg

        # Double check that we can load data from mongo even if we cannot make it
        self.st.context_config['forbid_creation_of'] = self.all_targets
        peaks = self.st.get_array(self.test_run_id, self.mongo_target)
        assert len(peaks)

    def test_write_and_change_lineage(self):
        """
        Lineage changes should result in data not being available
        and therefore the data should not be returned.
        """
        if not self.run_test:
            return
        self._make_mongo_target()
        assert self.is_stored_in_mongo

        # Check that lineage changes result in non-loadable data
        self.st.context_config['forbid_creation_of'] = self.all_targets
        self.st._plugin_class_registry['peaks'].__version__ = 'some other version'
        assert not self.is_stored_in_mongo
        with self.assertRaises(strax.DataNotAvailable):
            self.st.get_array(self.test_run_id, self.mongo_target)

    def test_clean_cache(self):
        """
        We keep a small cache in the backend of the last loaded data for
        offloading the database, test that it works
        """
        if not self.run_test:
            return
        self._make_mongo_target()
        assert self.is_stored_in_mongo
        mongo_backend = self.mongo_sf.backends[0]
        assert len(mongo_backend.chunks_registry) == 0, "nothing should be cached"
        # Now loading data should mean we start caching something
        self.st.get_array(self.test_run_id, self.mongo_target)
        len_before = len(mongo_backend.chunks_registry)
        assert len_before
        mongo_backend._clean_first_key_from_registry()
        assert len(mongo_backend.chunks_registry) < len_before

    def test_interrupt_iterator(self):
        """
        When we interrupt during the writing of data, make sure
        we are not able to data that is only half computed
        """
        if not self.run_test:
            return
        assert not self.is_stored_in_mongo
        self.st.config['n_chunks'] = 2  # Make sure that after one iteration we haven't finished
        for chunk in self.st.get_iter(self.test_run_id, self.mongo_target):
            print(chunk)
            break
        assert not self.is_stored_in_mongo

    def _make_mongo_target(self):
        assert not self.is_stored_in_mongo
        self.st.make(self.test_run_id, self.mongo_target)
        assert self.is_stored_in_mongo
