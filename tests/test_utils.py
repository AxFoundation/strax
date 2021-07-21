import numpy as np
import hypothesis.strategies as hst
from hypothesis import given, settings
import strax


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
def test_keep_columns(d):
    """
    Test that the keep_columns option of apply selection works. Also
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
