import strax
import numpy as np
from hypothesis import given, settings, strategies


def get_int_array(min_value=0,
                  max_value=1,
                  min_size=0,
                  max_size=20) -> strategies.lists:
    """
    Get array with ints
    :param min_value: min value of items in array
    :param max_value: max value of items in array
    :param min_size: min number of samples in array
    :param max_size: max number of samples in array
    :return: strategies.lists of integers of specified format
    """
    return strategies.lists(
        strategies.integers(min_value=min_value,
                            max_value=max_value),
        min_size=min_size,
        max_size=max_size)


def get_float_array(min_value=0, max_value=1, min_size=0, max_size=20):
    """
    Get array with floats
    :param min_value: min value of items in array
    :param max_value: max value of items in array
    :param min_size: min number of samples in array
    :param max_size: max number of samples in array
    :return: strategies.lists of floats of specified format
    """
    return strategies.lists(
        strategies.floats(min_value=min_value,
                          max_value=max_value),
        min_size=min_size,
        max_size=max_size)


@given(get_float_array(),
       get_int_array(max_value=100),
       get_float_array(min_size=20, max_size=150, max_value=100),
       )
@settings(deadline=None)
def test_local_minimum(min_heights, min_ratios, w):
    """
    see _test_splitter_inner
    """
    _test_splitter_inner(min_heights, min_ratios, w, 'natural_breaks')


@given(get_float_array(),
       get_int_array(max_value=100),
       get_float_array(min_size=20, max_size=150, max_value=100),
       )
@settings(deadline=None)
def test_natural_breaks(min_heights, min_ratios, w):
    """
    see _test_splitter_inner
    """
    _test_splitter_inner(min_heights, min_ratios, w, 'local_minimum')


def _test_splitter_inner(min_heights,
                         min_ratios,
                         waveform,
                         splitter):
    """
    Test the specified splitting algorithm
    :param min_heights: list of the minimum heights of the peaks to have a split
    :param min_ratios: list of the ratios of the peaks to have a split
    :param waveform: list (will be converted to array) of
    :param splitter: either 'local_minimum' or 'natural_breaks'
    """
    test_splitter = {
        'local_minimum': strax.processing.peak_splitting.LocalMinimumSplitter(),
        'natural_breaks': strax.processing.peak_splitting.NaturalBreaksSplitter()
                     }.get(splitter, None)
    print(f'Testing {splitter}')
    if test_splitter is None:
        raise NotImplementedError(f'Unknown splitter {splitter}')

    NO_MORE_SPLITS = strax.processing.peak_splitting.NO_MORE_SPLITS

    # mimick a peak
    waveform = np.array(waveform)

    for min_height, min_ratio in zip(min_heights, min_ratios):
        # Split according to the different splitters
        if splitter == 'local_minimum':
            my_splits = test_splitter.find_split_points(
                waveform, dt=None, peak_i=None, min_height=min_height,
                min_ratio=min_ratio)
        elif splitter == 'natural_breaks':
            # Use min-height here as threshold (>1 meaningless)
            threshold = np.array([min_height])
            my_splits = test_splitter.find_split_points(
                waveform, dt=1, peak_i=np.int(0), threshold=threshold, normalize=0,
                split_low=0, filter_wing_width=0)

        my_splits = np.array(list(my_splits))

        assert len(my_splits) >= 1
        # get left and right from found splits
        split_checks = [(int(split - 1), int(split + 1), int(split))
                        for split in my_splits[:, 0]]

        # discard last two split-entries if they exist
        # they are len(w) and NO_MORE_SPLITS --> nothing to test
        split_checks = split_checks[:-2]

        # This test does not have to work for the natural breaks
        # algorithm as we use a moving average
        if test_splitter == 'local_minimum':
            # check if left and right from split index value is bigger or equal
            for left, right, split in split_checks:
                assert waveform[left] >= waveform[split]
                assert waveform[right] >= waveform[split]

        assert len(my_splits) <= int(len(waveform) / 2) + 1
        assert min(my_splits[:, 0]) == NO_MORE_SPLITS
        assert my_splits[-1, 0] == NO_MORE_SPLITS
