import strax
import numpy as np


def test_local_minimum_splitter(splitter='local_minimum'):
    test_splitter = {
        'local_minimum': strax.processing.peak_splitting.LocalMinimumSplitter(),
        'natural_breaks': strax.processing.peak_splitting.NaturalBreaksSplitter()
                     }.get(splitter, None)
    print(f'Testing {splitter}')
    if test_splitter is None:
        raise NotImplementedError(f'Unknown splitter {splitter}')

    NO_MORE_SPLITS = strax.processing.peak_splitting.NO_MORE_SPLITS

    # arbitrary settings to check against
    min_heights = [0, 0.5, 1, 1.5]
    min_ratios = [0, 1, 10, 100]

    # mimick a peak
    w = np.random.random(size=100)

    for min_height, min_ratio in zip(min_heights, min_ratios):
        # Split according to the different splitters
        if splitter == 'local_minimum':
            my_splits = test_splitter.find_split_points(
                w, dt=None, peak_i=None, min_height=min_height,
                min_ratio=min_ratio)
        elif splitter == 'natural_breaks':
            # Use min-height here as threshold (>1 meaningless)
            threshold = np.array([min_height])
            my_splits = test_splitter.find_split_points(
                w, dt=1, peak_i=np.int(0), threshold=threshold, normalize=0,
                split_low=0, filter_wing_width=0)

        my_splits = np.array(list(my_splits))

        assert len(my_splits) >= 1
        # get left and right from found splits
        split_checks = [(int(split - 1), int(split + 1), int(split))
                        for split in my_splits[:, 0]]

        # discard last two split-entries if they exist
        # they are len(w) and NO_MORE_SPLITS --> nothing to test
        split_checks = split_checks[:-2]

        # check if left and right from split index value is bigger or equal
        for left, right, split in split_checks:
            assert min(w[split], w[left]) == w[split]
            assert min(w[split], w[right]) == w[split]

        assert len(my_splits) <= int(len(w) / 2) + 1
        assert min(my_splits[:, 0]) == NO_MORE_SPLITS
        assert my_splits[-1, 0] == NO_MORE_SPLITS


def test_natural_breaks():
    test_local_minimum_splitter(splitter='natural_breaks')
