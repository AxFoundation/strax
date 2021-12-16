import strax
import numpy as np
from scipy.stats import norm
from strax.processing.hitlets import highest_density_region_width


def test_highest_density_region():
    """
    Unity test for highest density regions.
    """
    # Some distribution:
    distribution = np.array([0, 0, 3, 4, 2, 0, 1])
    # Truth dict always stores fraction desired, intervals:
    truth_dict = {0.2: [[2, 4]], 0.7: [[2, 5], [6, 7]]}
    _test_highest_density_region(distribution, truth_dict)

    # Distribution with an offset:
    distribution = np.array([0, 0, 3, 4, 2, 0, 1]) + 2
    truth_dict = {0.2: [[2, 5]], 0.7: [[0, len(distribution)]]}
    _test_highest_density_region(distribution, truth_dict)


def _test_highest_density_region(distribution, truth_dict):
    intervals, heights = strax.highest_density_region(distribution,
                                                      np.array(list(truth_dict.keys())),
                                                      only_upper_part=True,
                                                      )
    for fraction_ind, (key, values) in enumerate(truth_dict.items()):
        for ind_interval, interval in enumerate(values):
            int_found = intervals[fraction_ind, :, ind_interval]
            mes = f'Have not found the correct edges for a fraction of {key}% found {int_found}, but expected ' \
                  f'{interval}'
            assert np.all(int_found == interval), mes


def test_too_small_buffer():
    """
    Unit test to check whether a too small buffer leads to np.nans
    """
    distribution = np.ones(1000)
    distribution[::4] = 0
    indicies, _ = strax.highest_density_region(distribution,
                                               np.array([0.5]),
                                               only_upper_part=True,
                                               )
    assert np.all(indicies == -1)

    width = highest_density_region_width(distribution,
                                         fractions_desired=np.array([0.5]),
                                         _buffer_size=10)
    assert np.all(np.isnan(width))


def test_true_hdr():
    """
    Tests if highest density region returns for a normal
    distribution the expected -1/1 boundaries for 68.27% coverage.

    We are not using a too high precision here to reduce the total test
    time.
    """
    x = np.arange(-5, 5, 10**-4)
    data = norm.pdf(x)
    data /= np.sum(data)
    index, _ = strax.highest_density_region(data, fractions_desired=np.array([0.6827]))
    a_index = index[0, 0, 0]
    b_index = index[0, 1, 0]
    area = np.sum(data[a_index:(b_index-1)])

    assert np.isclose(area, 0.6827, rtol=10**-4), (area, 0.6827)
    assert np.isclose(x[a_index], -1, rtol=10**-3), (x[a_index], -1)
    assert np.isclose(x[b_index-1], 1, rtol=10**-3), (x[b_index-1], 1)
