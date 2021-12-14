import numpy as np
import tempfile
import unittest
import os
import shutil
import hypothesis.strategies as hst
from hypothesis import given, settings
import strax
from strax.testutils import Records, Peaks, run_id


def test_growing_result():
    @strax.growing_result(np.int64, chunk_size=2)
    def bla(_result_buffer=None, result_dtype=None):
        buffer = _result_buffer
        offset = 0

        for i in range(5):
            buffer[offset] = i

            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0
        yield offset

    result = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    np.testing.assert_equal(bla(), result)
    should_get = result.astype(np.float64)
    got = bla(result_dtype=np.float64)
    np.testing.assert_equal(got, should_get)
    assert got.dtype == should_get.dtype


@hst.composite
def get_dummy_data(draw,
                   data_length=(0, 10),
                   dt=(1, 10),
                   max_time=(1, 20)
                   ):
    """
    Create some dummy array for testing apply selection. Data may be
        overlapping.

    :param data_length: desired length of the data
    :param dt: duration of each sample
    :return: data
    """
    # Convert ranges to int
    data_length = draw(hst.integers(*data_length))
    dt = draw(hst.integers(*dt))
    max_time = draw(hst.integers(*max_time))

    data = np.zeros(data_length, dtype=strax.time_fields + [('data', np.float64)])
    data['time'] = np.random.randint(0, max_time + 1, data_length)
    data['endtime'] = data['time'] + dt
    data['data'] = np.random.random(data_length)
    data.sort(order='time')
    return data


@settings(deadline=None)
@given(get_dummy_data(),
       hst.integers(0, 20),
       hst.integers(1, 10),
       )
def test_time_selection(d, second_time, second_dt):
    """
    Test that both 'touching' and 'fully_contained' give the same
        results as 'strax.fully_contained_in' and
        'strax.touching_windows' respectively
    :param d: test-data from get_dummy_data
    :param second_dt: the ofset w.r.t. the first
    :return: None
    """
    container = np.zeros(1, dtype=strax.time_fields)
    container['time'] = second_time
    container['endtime'] = second_time + second_dt
    time_range = (second_time, second_time + second_dt)

    # Fully contained in
    selected_data = strax.apply_selection(d,
                                          time_range=time_range,
                                          time_selection='fully_contained')
    contained = strax.fully_contained_in(d, container)
    selected_data_fc = d[contained != -1]
    assert np.all(selected_data == selected_data_fc)

    # TW
    selected_data = strax.apply_selection(d,
                                          time_range=time_range,
                                          time_selection='touching')
    windows = strax.touching_windows(d, container, window=0)
    assert np.diff(windows[0]) == len(selected_data)
    if len(windows) and len(selected_data):
        assert np.all(selected_data == d[windows[0][0]:windows[0][1]])


@settings(deadline=None)
@given(get_dummy_data(
    data_length=(1, 10),
    dt=(1, 10),
    max_time=(1, 20)))
def test_selection_str(d):
    """
    Test selection string. We are going for this example check that
        selecting the data based on the data field is the same as if we
        were to use a mask NB: data must have some length!

    :param d: test-data from get_dummy_data
    :return: None
    """
    mean_data = np.mean(d['data'])
    max_data = np.max(d['data'])
    mask = (d['data'] > mean_data) & (d['data'] < max_data)
    selections_str = [f'data > {mean_data}', 
                      f'data < {max_data}']
    selected_data = strax.apply_selection(d, selection_str=selections_str)
    assert np.all(selected_data == d[mask])


@settings(deadline=None)
@given(get_dummy_data(
    data_length=(0, 10),
    dt=(1, 10),
    max_time=(1, 20)),
)
def test_keep_drop_columns(d):
    """
    Test that the keep/drop_columns option of apply selection works. Also
        test that it does not affect the original array (e.g. if it were
        to use a view instead of a copy).

    :param d: test-data from get_dummy_data
    :return: None
    """
    columns = list(d.dtype.names)
    selected_data = strax.apply_selection(d, keep_columns=columns[1:])

    # Check we din't loose anything of the original array
    assert columns == list(d.dtype.names)

    for c in columns[1:]:
        assert np.all(selected_data[c] == d[c])
    for c in columns[:1]:
        assert c not in selected_data.dtype.names
    
    # Repeat test but for drop columns:
    selected_data = strax.apply_selection(d, drop_columns=columns[:1])
    for c in columns[1:]:
        assert np.all(selected_data[c] == d[c])
    for c in columns[:1]:
        assert c not in selected_data.dtype.names


class TestMultiRun(unittest.TestCase):
    """Test behavior of multi-runs for various different settings."""

    def setUp(self):
        """Setup context and make some subruns."""
        self.tempdir = tempfile.mkdtemp()
        self.context = strax.Context(storage=[strax.DataDirectory(self.tempdir,
                                                                  provide_run_metadata=True,
                                                                  readonly=False,
                                                                  deep_scan=True)],
                                     register=[Records, Peaks],
                                     config={'bonus_area': 42,
                                             'use_per_run_defaults': False,
                                             'recs_per_chunk': 10**3,
                                             },
                                     )
        self.run_ids = [str(r) for r in range(5)]

        for run_id in self.run_ids:
            self.context.make(run_id, 'records')

    def test_multi_run(self):
        self._test_get_array_multi_run()

    def test_multi_run_more_workers(self):
        self._test_get_array_multi_run(max_worker=2)

    def test_multi_run_multiprocessing(self):
        self.context.set_context_config({'allow_multiprocess': True})
        self._test_get_array_multi_run(max_worker=2)

    def test_multi_run_memory_profile(self):
        """Tests if multi-runs fills up memory. Use only a single worker
        hence there should not be at any time more than 2 runs inside the
        ThreadPoolExecutor.
        """
        from memory_profiler import memory_usage
        # First get overhead to set up computing:
        mem_overhead = memory_usage((self.context.make,
                                     ('1', 'records')))
        mem_overhead = np.mean(mem_overhead)

        # Now get size of a single array:
        size_per_run = self.context.size_mb('1', 'records')
        rr_test = self.context.get_array('1', 'records')
        # Make 50 runs and compare memory usage:
        used_mem = memory_usage((self.context.make,
                                 ([str(r) for r in range(50)], 'records'))
                                )
        peak_mem = np.mean(used_mem)

        # Be a bit more generous and put the limit to 5 runs. If we
        # really fill up the memory processing 50 runs should show it.
        assert peak_mem <= mem_overhead + size_per_run*5

    def _test_get_array_multi_run(self, max_worker=None):
        rr = []
        for run_id in self.run_ids:
            rr.append(self.context.get_array(run_id, 'records'))
        rr = np.concatenate(rr)

        rr_multi = self.context.get_array(self.run_ids, 'records', max_worker=max_worker)
        assert len(rr) == len(rr_multi)
        assert np.all(rr['time'] == rr_multi['time'])

        # Shuffle run_ids and check if output is sorted:
        np.random.shuffle(self.run_ids)
        rr_multi = self.context.get_array(self.run_ids, 'records', max_worker=max_worker)
        assert np.all(np.diff(rr_multi['run_id'].astype(np.int8)) >= 0)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
