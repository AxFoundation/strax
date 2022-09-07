from unittest import TestCase
import numpy as np
import strax


# class ContiniousData(strax.Plugin):
#     n_chunks = strax.Config(type=int, default=10, track=False)
#     recs_per_chunk = strax.Config(type=int, default=10, track=False)
#     time_per_rec = strax.Config(type=int, default=2, track=False)
#
#     provides = 'records'
#     parallel = 'process'
#     depends_on = tuple()
#     dtype = strax.time_fields
#
#     rechunk_on_save = False
#
#     def source_finished(self):
#         return True
#
#     def is_ready(self, chunk_i):
#         return chunk_i < self.n_chunks
#
#     def compute(self, chunk_i):
#         r = np.zeros(self.config['recs_per_chunk'], self.dtype)
#         r['time'] = chunk_i * np.arange(self.recs_per_chunk) * self.time_per_rec
#
#         r['endtime'] = r['time'] + self.time_per_rec
#         return self.chunk(start=r['time'][1], end=r['endtime'][-1], data=r)


class RandomTimeData(strax.RechunkerPlugin):
    chunk_order = strax.Config(default=list(np.arange(10)[::-1]), track=False)
    recs_per_chunk = strax.Config(default=list(np.arange(1, 11)), track=False)
    time_per_rec = strax.Config(type=int, default=2, track=False)


    provides = 'random_records'
    parallel = 'process'
    depends_on = tuple()
    dtype = strax.time_fields

    rechunk_on_save = False

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < len(self.chunk_order)

    def compute(self, chunk_i):
        ith_order = self.chunk_order[chunk_i]
        n_recs = self.recs_per_chunk[chunk_i]
        r = np.zeros(n_recs, self.dtype)
        r['time'] = ith_order * np.arange(n_recs) * self.time_per_rec

        r['endtime'] = r['time'] + self.time_per_rec
        return self.chunk(start=r['time'][0], end=r['endtime'][-1], data=r)


class Counter(strax.Plugin):
    provides='counts'
    depends_on='random_records'
    data_kind = 'counts'
    dtype = strax.time_fields + [(('counts per chunk', 'counts'),  np.int64)]

    def compute(self, random_records):
        res = np.zeros(1, self.dtype)
        res['time'] = random_records['time'].min()
        res['endtime'] = random_records['endtime'].max()
        res['counts'] = len(random_records)
        return res



class TestRandomOrder(TestCase):
    def setUp(self) -> None:
        st = strax.Context(storage=[])
        st.register(RandomTimeData)
        st.register(Counter)
        self.run_id = '0'
        self.st = st

    def test_make(self):
        st = self.st
        # st.make(self.run_id, RandomTimeData.provides)
        st.make(self.run_id, Counter.provides)


# TOTAL_DEADTIME = 0
#
# class DummyAqmonHits(strax.Plugin):
#     # Strax does not like numpy
#     vetos_per_chunk=strax.Config(default=[int(n) for n in np.arange(10, 13)])
#     start_with_channel_on=strax.Config(default=True)
#     channel_on=strax.Config(
#         default=1,
#         type=int)
#     channel_off=strax.Config(
#         default=2,
#         type=int)
#     dt_max = strax.Config(default=1e9)
#     depends_on = ()
#     parallel = False
#     provides = straxen.AqmonHits.provides
#     dtype = straxen.AqmonHits.dtype
#     save_when = strax.SaveWhen.NEVER
#     _last_channel_was_off = True
#     _last_endtime = 0
#
#     def source_finished(self):
#         return True
#
#     def is_ready(self, chunk_i):
#         return chunk_i < len(self.vetos_per_chunk)
#
#     def compute(self, chunk_i):
#         if chunk_i == 0:
#             self._last_channel_was_off = self.start_with_channel_on
#         n_vetos = self.vetos_per_chunk[chunk_i]
#
#         res = np.zeros(n_vetos, self.dtype)
#
#         # Add some randomly increasing times (larger than previous chunk
#         res['time'] = self._last_endtime + 1 + np.cumsum(
#             np.random.randint(low=2, high=self.dt_max, size=n_vetos)
#         )
#         res['dt'] = 1
#         res['length'] = 1
#
#         if self._last_channel_was_off:
#             res['channel'][::2] = self.channel_on
#             res['channel'][1::2]= self.channel_off
#             starts = res[::2]['time']
#             stops = res[1::2]['time']
#             self.TOTAL_DEADTIME += [np.sum(stops - starts[:len(stops)])]
#
#         else:
#             res['channel'][1::2] = self.channel_on
#             res['channel'][::2] = self.channel_off
#             starts = res[1::2]['time']
#             stops = res[::2]['time']
#
#             # Ignore the first stop (stops last from previous chunk)
#             self.TOTAL_DEADTIME += [np.sum(stops[1:] - starts[:len(stops)-1])]
#
#             # Additionally, add the deadtime that spans the chunks
#             self.TOTAL_DEADTIME += [stops[0]-self._last_endtime]
#
#         self.TOTAL_SIGNALS += [len(res)]
#
#         previous_end = self._last_endtime
#         self._last_endtime = res['time'][-1]
#         self._last_channel_was_off = res['channel'][-1] == self.channel_off
#         if len(res)>1:
#             assert np.sum(res['channel'])
#         print(self._last_endtime)
#         return self.chunk(start=previous_end+1,
#                           end=strax.endtime(res)[-1],
#                           data=res)
#
#
# class TestAqmonProcessing(TestCase):
#     def setUp(self) -> None:
#         st = straxen.test_utils.nt_test_context().new_context()
#         st._plugin_class_registry = {}
#         st.set_config(dict(veto_proximity_window=10**99))
#         self.TOTAL_DEADTIME = []
#         self.TOTAL_SIGNALS = []
#
#         class DeadTimedDummyAqHits(DummyAqmonHits):
#             TOTAL_DEADTIME = self.TOTAL_DEADTIME
#             TOTAL_SIGNALS = self.TOTAL_SIGNALS
#
#         class DummyVi(straxen.acqmon_processing.VetoIntervals):
#             save_when = strax.SaveWhen.NEVER
#
#         class DummyVp(straxen.acqmon_processing.VetoIntervals):
#             save_when = strax.SaveWhen.NEVER
#
#         st.register(DeadTimedDummyAqHits)
#         st.register(DummyVi)
#         st.register(DummyVp)
#         self.st = st
#         self.run = '999997'
#         assert not np.sum(self.TOTAL_DEADTIME)
#         assert not st.is_stored(self.run, 'aqmon_hits')
#
#     def test_dummy_plugin_works(self):
#         self.st.make(self.run, 'aqmon_hits')
#         assert np.sum(self.TOTAL_DEADTIME)
#         assert self.TOTAL_SIGNALS
#
#     def test_veto_intervals(self):
#         assert not self.st.is_stored(self.run, 'aqmon_hits')
#         assert not self.st.is_stored(self.run, 'veto_intervals')
#         veto_intervals = self.st.get_array(self.run, 'veto_intervals')
#         assert len(veto_intervals)
#         assert np.sum(self.TOTAL_DEADTIME)
#         assert np.sum(veto_intervals['veto_interval']) == np.sum(self.TOTAL_DEADTIME)