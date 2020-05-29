import strax
import numpy as np


def test_LocalMinimumSplitter():
    test_splitter = strax.processing.peak_splitting.LocalMinimumSplitter()
    NO_MORE_SPLITS = strax.processing.peak_splitting.NO_MORE_SPLITS

    # arbitrary settings to check against
    min_heights = [0, 0.5, 1, 1.5]
    min_ratios = [0, 1, 10, 100]

    # mimick a peak
    w = np.random.random(size=100)

    for min_height, min_ratio in zip(min_heights, min_ratios):
        my_splits = test_splitter.find_split_points(w,
                                                    dt=None,
                                                    peak_i=None,
                                                    min_height=min_height,
                                                    min_ratio=min_ratio)

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
