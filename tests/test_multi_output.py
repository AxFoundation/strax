import tempfile

import strax
from strax.testutils import *
from immutabledict import immutabledict
import pytest

import unittest
import os
import shutil


class EvenOddSplit(strax.Plugin):
    parallel = 'process'
    depends_on = 'records'
    provides = ('even_recs', 'odd_recs', 'rec_count')

    data_kind = dict(
        even_recs='even_recs',
        odd_recs='odd_recs',
        rec_count='chunk_bonus_data')

    dtype = dict(
        even_recs=Records.dtype,
        odd_recs=Records.dtype,
        rec_count=[
            ('time', np.int64, 'Time of first record in chunk'),
            ('endtime', np.int64, 'Endtime of last record in chunk'),
            ('n_records', np.int32, 'Number of records'),
        ])

    def compute(self, records):
        mask = records['time'] % 2 == 0
        return dict(even_recs=records[mask],
                    odd_recs=records[~mask],
                    rec_count=dict(n_records=[len(records)],
                                   time=records[0]['time'],
                                   endtime=strax.endtime(records[-1])))


class ZipRecords(strax.Plugin):
    provides = 'zipped_records'
    depends_on = ('even_recs', 'odd_recs')
    data_type = 'zipped_records'
    dtype = Records.dtype
    parallel = True

    def compute(self, even_recs, odd_recs):
        return strax.sort_by_time(np.concatenate((even_recs, odd_recs)))


class EvenRecsClassified(strax.Plugin):
    """
    Plugin required to test inline plugins for double dependency.
    """
    provides = 'even_recs_classified'
    depends_on = 'even_recs' 
    parallel = True
    
    def infer_dtype(self):
        dtype = []
        dtype += strax.time_dt_fields
        dtype += [(('dummy classification', 'classifier'), np.int8)]
        return dtype
   
    def compute(self, even_recs):
        res = np.ones(len(even_recs), self.dtype)
        res['time'] = even_recs['time']
        res['dt'] = even_recs['dt']
        res['length'] = even_recs['length']
        return res


class ZipRecordsAdditionalDependency(strax.Plugin):
    """
    Plugin required to test inline plugins for double dependency.
    """
    provides = 'zipped_records_classified'
    depends_on = ('even_recs', 'even_recs_classified', 'odd_recs')
    data_type = 'zipped_records_classified'
    dtype = Records.dtype
    parallel = True

    def compute(self, even_recs, odd_recs):
        n_even = len(even_recs)
        n_odd = len(odd_recs)
        res = np.zeros(n_even+n_odd, self.dtype)
        res['dt'][:n_even] = even_recs['dt']
        res['time'][:n_even] = even_recs['time']
        res['length'][:n_even] = even_recs['length']
        res['channel'][:n_even] = even_recs['channel']
        
        res['dt'][n_even:] = odd_recs['dt']
        res['time'][n_even:] = odd_recs['time']
        res['length'][n_even:] = odd_recs['length']
        res['channel'][n_even:] = odd_recs['channel']
        
        return strax.sort_by_time(res)


class FunnyPeaks(strax.Plugin):
    parallel = True
    provides = 'peaks'
    depends_on = 'even_recs'
    dtype = strax.peak_dtype()

    def compute(self, even_recs):
        p = np.zeros(len(even_recs), self.dtype)
        p['time'] = even_recs['time']
        return p


class TestMultiOutputs(unittest.TestCase):

    def setUp(self):
        # Temp directory for storing record data for the tests.
        # Will be removed during TearDown.
        self.temp_dir = tempfile.mkdtemp()
        self.mystrax = strax.Context(
            storage=strax.DataDirectory(self.temp_dir),
            register=[Records, EvenOddSplit, ZipRecords, FunnyPeaks, 
                      EvenRecsClassified, ZipRecordsAdditionalDependency],
            allow_multiprocess=True)
        assert not self.mystrax.is_stored(run_id, 'records')
        assert not self.mystrax.is_stored(run_id, 'rec_count')

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Set all plugins be back to defaul SaveWhen.ALWAYS,
        # somehow the change remain otherwise.
        for p in self.mystrax._plugin_class_registry.values():
            p.save_when = strax.SaveWhen.ALWAYS

    def test_save_when_per_provide_same_save_when(self):
        """
        Tests if we can specify the save_when parameter per data_type
        provided by the plugin.
        """
        # First check if default save_when is converted into a dict
        # and that all values are correct:
        p = self.mystrax.get_single_plugin('0', 'even_recs')
        assert isinstance(p.save_when, immutabledict)
        assert np.all([d in p.save_when for d in p.provides])
        assert np.all([p.save_when[d] == strax.SaveWhen.ALWAYS for d in p.provides])

        # Test if data is stored correctly:
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'odd_recs')
        assert not self.mystrax.is_stored('0', 'rec_count')
        self.mystrax.make('0', 'even_recs')
        assert self.mystrax.is_stored('0', 'even_recs')
        assert self.mystrax.is_stored('0', 'odd_recs')
        assert self.mystrax.is_stored('0', 'rec_count')

    def test_save_when_per_provide(self):
        """
        Tests if save when works properly in case of different save when
        per provided data_type.
        """
        _save_when = immutabledict({'even_recs': strax.SaveWhen.NEVER,
                                    'odd_recs': strax.SaveWhen.TARGET,
                                    'rec_count': strax.SaveWhen.ALWAYS,
                                    })
        p = self.mystrax._plugin_class_registry['rec_count']
        p.save_when = _save_when

        p = self.mystrax.get_single_plugin('0', 'even_recs')
        for d in p.provides:
            assert p.save_when[d] == _save_when[d]

        # Test if call of NEVER data_type makes any data:
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'odd_recs')
        assert not self.mystrax.is_stored('0', 'rec_count')
        self.mystrax.make('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'odd_recs')
        assert not self.mystrax.is_stored('0', 'rec_count')

        # See if only ALWAYS is made:
        self.mystrax.make('0', 'rec_count')
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'odd_recs')
        assert self.mystrax.is_stored('0', 'rec_count')

        # Also check if time_range selection only work for already made
        # data:
        # SaveWhen.NEVER and ALWAYS should work:
        res = self.mystrax.get_array('0', 'even_recs', time_range=(0, 1))
        assert len(res), res
        res = self.mystrax.get_array('0', 'rec_count', time_range=(0, 1))
        assert len(res), res
        with pytest.raises(strax.DataNotAvailable):
            self.mystrax.get_array('0', 'odd_recs', time_range=(0, 1))

        # Check if saver of already existing data dropped:
        components = self.mystrax.get_components('0', 'odd_recs')
        assert 'odd_recs' in components.savers
        assert 'rec_count' not in components.savers

        # See if TARGET is made and previous always does not raise error
        # since it already exists:
        self.mystrax.make('0', 'odd_recs')
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert self.mystrax.is_stored('0', 'odd_recs')
        assert self.mystrax.is_stored('0', 'rec_count')

    def test_save_per_provide_inlined(self):
        """
        Checks whether the plugin inlining works for different
        combinations.
        """
        _save_when = immutabledict({'even_recs': strax.SaveWhen.NEVER,
                                    'odd_recs': strax.SaveWhen.TARGET,
                                    'rec_count': strax.SaveWhen.ALWAYS,
                                    })
        p = self.mystrax._plugin_class_registry['rec_count']
        p.save_when = _save_when

        p = self.mystrax._plugin_class_registry['even_recs_classified']
        p.save_when = strax.SaveWhen.EXPLICIT

        p = self.mystrax.get_single_plugin('0', 'even_recs')
        for d in p.provides:
            assert p.save_when[d] == _save_when[d]

        # Test inlining with excluded steps:
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'even_recs_classified')
        assert not self.mystrax.is_stored('0', 'peaks')
        self.mystrax.make('0', 'peaks')
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert not self.mystrax.is_stored('0', 'even_recs_classified')
        assert self.mystrax.is_stored('0', 'peaks')

        # Test if saving EXPLICIT works:
        self.mystrax.make('0', 'even_recs_classified', save=('even_recs_classified', ))
        assert not self.mystrax.is_stored('0', 'even_recs')
        assert self.mystrax.is_stored('0', 'even_recs_classified')

    def test_double_dependency(self):
        """
        Tests if double dependency of a plugin on another plugin leads
        to dead lock in processing.
        """
        self.mystrax.set_context_config({'timeout': 120})  # Set time out to 2 min
        self._test_double_dependency()
        
    def test_double_dependency_notlazy(self):
        """
        Tests if double dependency of a plugin on another plugin leads
        to dead lock in processing.
        """
        self.mystrax.set_context_config({'timeout': 120, 
                                         'allow_lazy': False})  
        self._test_double_dependency()
        
    def test_double_dependency_multiprocess(self):
        """
        Tests if double dependency of a plugin on another plugin leads
        to dead lock in processing.
        """
        self.mystrax.set_context_config({'timeout': 120, 
                                         'allow_lazy': False,
                                         'allow_multiprocess': True})  
        self._test_double_dependency(max_workers=2)
        
    def test_double_dependency_inline_plugins(self):
        """
        Tests if double dependency of a plugin on another plugin leads
        to dead lock in processing.
        """
        self.mystrax.set_context_config({'timeout': 120, 
                                         'allow_lazy': False,
                                         'allow_multiprocess': True})  
        self._test_double_dependency(max_workers=2, make_data_type='zipped_records_classified')
        self.mystrax.is_stored(run_id, 'even_recs_classified')

    def test_multi_output(self):
        for max_workers in [1, 2]:
            # Create stuff
            funny_ps = self.mystrax.get_array(
                run_id=run_id,
                targets='peaks',
                max_workers=max_workers)
            p = self.mystrax.get_single_plugin(run_id, 'records')
            assert self.mystrax.is_stored(run_id, 'peaks')
            assert self.mystrax.is_stored(run_id, 'even_recs')

            # Peaks are correct
            assert np.all(funny_ps['time'] % 2 == 0)
            assert len(funny_ps) == p.config['n_chunks'] * p.config['recs_per_chunk'] / 2

            # Unnecessary things also got stored
            assert self.mystrax.is_stored(run_id, 'rec_count')
            assert self.mystrax.is_stored(run_id, 'odd_recs')

            # Record count is correct
            rec_count = self.mystrax.get_array(run_id, 'rec_count')
            print(rec_count)
            assert len(rec_count) == p.config['n_chunks']
            np.testing.assert_array_equal(rec_count['n_records'], p.config['recs_per_chunk'])

            # Even and odd records are correct
            r_even = self.mystrax.get_array(run_id, 'even_recs')
            r_odd = self.mystrax.get_array(run_id, 'odd_recs')
            assert np.all(r_even['time'] % 2 == 0)
            assert np.all(r_odd['time'] % 2 == 1)
            assert len(r_even) == p.config['n_chunks'] * p.config['recs_per_chunk'] / 2
            assert len(r_even) == len(r_odd)

    def _test_double_dependency(self, make_data_type='zipped_records', **kwargs):
        assert not self.mystrax.is_stored(run_id, 'even_recs')
        assert not self.mystrax.is_stored(run_id, 'odd_recs')
        
        self.mystrax.make(run_id, 'records')
        zipped_records = self.mystrax.get_array(run_id, make_data_type, **kwargs)
        records = self.mystrax.get_array(run_id, 'records')
        assert np.all(zipped_records == records)
        assert self.mystrax.is_stored(run_id, 'even_recs')
        assert self.mystrax.is_stored(run_id, 'odd_recs')
        assert self.mystrax.is_stored(run_id, make_data_type)
