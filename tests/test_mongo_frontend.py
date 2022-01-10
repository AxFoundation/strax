import unittest
import strax
from strax.testutils import Records, Peaks
import os
import pymongo
import logging
import time
_can_test = 'TEST_MONGO_URI' in os.environ


@unittest.skipIf(not _can_test, 'No test-database is configured')
class TestMongoFrontend(unittest.TestCase):
    """
    Test the saving behavior of the context with the strax.MongoFrontend

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.
    """

    def setUp(self):
        self.test_run_id = '0'
        self.all_targets = ('peaks', 'records')
        self.mongo_target = 'peaks'

        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test_mongosf_database'
        self.collection_name = 'temp-test-collection-mongosf'
        client = pymongo.MongoClient(uri)
        self.database = client[db_name]
        self.collection.drop()
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
        self.log = logging.getLogger(self.__class__.__name__)
        assert not self.is_all_targets_stored

    def tearDown(self):
        self.collection.drop()

    @property
    def collection(self):
        return self.database[self.collection_name]

    @property
    def is_all_targets_stored(self) -> bool:
        """This should always be False as one of the targets (records) is not stored in mongo"""
        return all([self.st.is_stored(self.test_run_id, t) for t in self.all_targets])

    def stored_in_context(self, context):
        return context._is_stored_in_sf(self.test_run_id, self.mongo_target, self.mongo_sf)

    @property
    def is_stored_in_mongo(self) -> bool:
        return self.stored_in_context(self.st)

    @property
    def is_data_in_collection(self):
        return self.collection.find_one() is not None

    def test_write_and_load(self):
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
        assert not self.is_stored_in_mongo
        self.st.config['n_chunks'] = 2  # Make sure that after one iteration we haven't finished
        for chunk in self.st.get_iter(self.test_run_id, self.mongo_target):
            print(chunk)
            break
        assert not self.is_stored_in_mongo

    def test_allow_incomplete(self):
        """Test loading incomplete data"""
        st_incomplete_allowed = self.st.new_context()
        st_incomplete_allowed.set_context_config(
            {'allow_incomplete': True,
             'forbid_creation_of': '*',
             }
        )
        assert not self.is_stored_in_mongo
        self.st.config['n_chunks'] = 3

        self.log.info(f'Starting with empty db {self.chunk_summary}')
        # Get the iterator separately and complete with "next(iterator)
        iterator = self.st.get_iter(self.test_run_id, self.mongo_target)

        self.log.info(f'Got iterator, still no data?: {self.chunk_summary}')
        # Chunk 0
        if self.is_stored_in_mongo:
            raise RuntimeError(f'Target should not be stored'
                               f'\n{self.chunk_summary}')

        # Chunk 1
        next(iterator)
        time.sleep(0.5)  # In case the database is not reflecting changes quick enough
        self.log.info(f'Got first chunk, still no data?: {self.chunk_summary}')
        if self.is_stored_in_mongo:
            raise RuntimeError(f'After 1 chunk target should not be stored'
                               f'\n{self.chunk_summary}')
        if not self.stored_in_context(st_incomplete_allowed):
            raise RuntimeError(f'We did not find the one chunk that should be '
                               f'allowed to be loaded\n {self.chunk_summary}')

        # Chunks >1
        for _ in iterator:
            pass

        stored_in_st = self.is_stored_in_mongo
        stored_in_incomplete_st = self.stored_in_context(st_incomplete_allowed)
        if not stored_in_st or not stored_in_incomplete_st:
            raise RuntimeError(f'Source finished and should be stored in st '
                               f'({stored_in_st}) and st_incomplete ('
                               f'{stored_in_incomplete_st}) '
                               f'\n{self.chunk_summary}')

    def test_allow_incomplete_during_md_creation(self):
        """
        Test that allowing incomplete data does not find data if the
        metadata ("md") is just created

        See #596 for more info

        Test is different from "test_allow_incomplete" in the sense
        that no chunks are written at all, only the md is registered
        """
        st_incomplete_allowed = self.st.new_context()
        st_incomplete_allowed.set_context_config(
            {'allow_incomplete': True,
             'forbid_creation_of': '*',
             }
        )
        assert not self.is_stored_in_mongo
        self.st.config['n_chunks'] = 3

        # Mimic metadata creation:
        # github.com/AxFoundation/strax/blob/a9ec08003a9193113c65910602d8b1b0ed4eb4e6/strax/context.py#L903  # noqa
        # Get the iterator separately and complete with "next(iterator)
        key = self.st.key_for(self.test_run_id, self.mongo_target)
        target_plugin = self.st.get_single_plugin(self.test_run_id, self.mongo_target)
        self.mongo_sf.saver(
            key=key,
            metadata=target_plugin.metadata(self.test_run_id, self.mongo_target),
            saver_timeout=self.st.context_config['saver_timeout'])
        print(self.chunk_summary)
        assert len(self.chunk_summary) == 1, f'Only md should be written {self.chunk_summary}'

        # Now check that both frontends understand there is no data (even when
        # allow_incomplete is set)
        stored_in_st = self.is_stored_in_mongo
        stored_in_incomplete_st = self.stored_in_context(st_incomplete_allowed)
        if stored_in_st or stored_in_incomplete_st:
            raise RuntimeError(f'Only metadata written and should NOT stored in st '
                               f'({stored_in_st}) and st_incomplete ('
                               f'{stored_in_incomplete_st}) '
                               f'\n{self.chunk_summary}')

    def _make_mongo_target(self):
        assert not self.is_stored_in_mongo
        self.st.make(self.test_run_id, self.mongo_target)
        assert self.is_stored_in_mongo

    @staticmethod
    def _return_file_info(file: dict,
                          save_properties=('number',
                                           'data_type',
                                           'lineage_hash',
                                           'provides_meta',
                                           'chunk_i',
                                           'chunks',
                                           )
                          ) -> dict:
        return {k: file.get(k) for k in save_properties}

    @property
    def chunk_summary(self):
        files = self.collection.find()
        return [self._return_file_info(f) for f in files]


if __name__ == '__main__':
    test = TestMongoFrontend()

    for attribute in test.__dict__.keys():
        if attribute.startswith('test_'):
            test.setUp()
            func = getattr(test, attribute)
            func()
            test.tearDown()
    print('Done bye bye')
