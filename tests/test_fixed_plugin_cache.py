from strax.testutils import Records, Peaks
import strax
import unittest
import numpy as np


class ChannelIsRunidRecords(Records):
    """Set the channel field equal to the run_id"""
    def compute(self, chunk_i):
        res = super().compute(chunk_i)
        res.data['channel'][:] = int(self.run_id)
        return res


class MaxChannelPeaks(Peaks):
    def infer_dtype(self):
        # We are going to check later that the infer_dtype is always called.
        dtype = strax.peak_dtype() + [
            (('PMT with median most records',
              'max_pmt'), np.int16)
        ]
        self.dtype_is_set = True
        return dtype

    def compute(self, records):
        assert np.all(records['channel'] == int(self.run_id))
        res = super().compute(records)
        res['max_pmt'] = records['channel'].mean()
        return res


class TestContextFixedPluginCache(unittest.TestCase):
    """Test the _fixed_plugin_cache of a context"""
    def test_load_runs(self, n_runs=3, config_update=None, **kwargs):
        """Try loading data for n_runs to make sure that we are """
        run_ids = [str(r) for r in range(n_runs)]
        st = self.get_context(use_per_run_defaults=False)
        if config_update is not None:
            st.set_context_config(config_update)
        data = st.get_array(run_ids, 'records', **kwargs)
        run_id_channel_diff = data['run_id'].astype(np.int64) - data['channel']
        assert np.all(run_id_channel_diff == 0)

        # To be sure also double check Peaks as self.deps of the Plugin
        # class should be correctly taken care of by the context.
        peaks_data = st.get_array(run_ids, 'peaks')
        run_id_max_pmt_diff = peaks_data['max_pmt'] - peaks_data['run_id'].astype(np.int64)
        assert np.all(run_id_max_pmt_diff == 0)

    def test_get_plugin(self, n_runs=3):
        run_ids = [str(r) for r in range(n_runs)]
        st = self.get_context(use_per_run_defaults=False)
        plugins_seen = []
        for run in run_ids:
            p = st.get_single_plugin(run, 'records')
            plugins_seen.append(p)
            assert p.run_id == run

        # If we passed around a reference instead of a copy of the
        # plugin, this would be a problem.
        for r_i, run in enumerate(run_ids):
            assert plugins_seen[r_i].run_id == run

    def test_load_runs_multicore(self):
        """Load the runs. If the references are mixed up the results are inconsistent"""
        multicore_config = dict(allow_lazy=False,
                                timeout=60,
                                allow_multiprocess=True,
                                )
        self.test_load_runs(n_runs=10, config_update=multicore_config, max_workers=10)

    def test_cache_changes(self):
        """
        Test that the _fixed_plugin_cache changes if we:
          - Change the config
          - Change the version of a plugin

         """
        st = self.get_context(use_per_run_defaults=False)

        # Compute the key/hash under which we will store the plugins
        first_key = st._context_hash()
        assert first_key is not None

        # Change the config triggers a new key
        st.set_config({'bla': 1})
        second_key = st._context_hash()

        # Change the version of a plugin triggers a new key
        st._plugin_class_registry['records'].__version__ = -1
        third_key = st._context_hash()

        assert first_key != second_key != third_key

    def test_set_dtype(self):
        st = self.get_context(use_per_run_defaults=False)

        # Compute the key/hash under which we will store the plugins
        st.key_for('0', 'peaks')
        assert st._fixed_plugin_cache[st._context_hash()]['peaks'].dtype_is_set

        # Now recreate for a new run
        st.key_for('1', 'peaks')
        assert st._fixed_plugin_cache[st._context_hash()]['peaks'].dtype_is_set

    @staticmethod
    def get_context(use_per_run_defaults: bool):
        """Get simple context"""
        st = strax.Context(storage=[],
                           register=(ChannelIsRunidRecords, MaxChannelPeaks),
                           config=dict(bonus_area=1))
        st.set_context_config({'use_per_run_defaults': use_per_run_defaults})
        return st
