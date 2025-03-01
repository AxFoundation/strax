import numpy as np
import numba
import strax

export, __all__ = strax.exporter()


@export
def split_peaks(
    peaks,
    hits,
    records,
    rlinks,
    to_pe,
    algorithm="local_minimum",
    data_type="peaks",
    n_top_channels=0,
    store_data_top=False,
    store_data_start=False,
    **kwargs,
):
    """Return peaks split according to algorithm, with waveforms summed and widths computed.

    Note:
        Can also be used for hitlets splitting with local_minimum
        splitter. Just put hitlets instead of peaks.

    :param peaks: Original peaks. Sum waveform must have been built
    and properties must have been computed (if you use them)
    :param hits: Hits found in records. (or None in case of hitlets
        splitting.)
    :param records: Records from which peaks were built
    :param rlinks: strax.record_links for given records
        (or None in case of hitlets splitting.)
    :param to_pe: ADC to PE conversion factor array (of n_channels)
    :param algorithm: 'local_minimum' or 'natural_breaks'.
    :param data_type: 'peaks' or 'hitlets'. Specifies whether to use
        sum_wavefrom or get_hitlets_data to compute the waveform of
        the new split peaks/hitlets.
    :param n_top_channels: Number of top array channels.
    :param result_dtype: dtype of the result.
    :param store_data_top: Boolean which indicates whether to store the top array
        waveform in the peak.
    :param store_data_start: Boolean which indicates whether to store the first samples of the
        waveform in the peak.

    Any other options are passed to the algorithm.

    """
    splitter = dict(local_minimum=LocalMinimumSplitter, natural_breaks=NaturalBreaksSplitter)[
        algorithm
    ]()

    data_type_is_not_supported = data_type not in ("hitlets", "peaks")
    if data_type_is_not_supported:
        raise TypeError(f'Data_type "{data_type}" is not supported.')
    return splitter(
        peaks,
        hits,
        records,
        rlinks,
        to_pe,
        data_type,
        n_top_channels=n_top_channels,
        store_data_top=store_data_top,
        store_data_start=store_data_start,
        **kwargs,
    )


NO_MORE_SPLITS = -9999999


class PeakSplitter:
    """Split peaks into more peaks based on arbitrary algorithm.

    :param peaks: Original peaks. Sum waveform must have been built and properties must have been
        computed (if you use them).
    :param records: Records from which peaks were built.
    :param rlinks: strax.record_links for given records.
    :param to_pe: ADC to PE conversion factor array (of n_channels).
    :param data_type: 'peaks' or 'hitlets'. Specifies whether to use sum_waveform or
        get_hitlets_data to compute the waveform of the new split peaks/hitlets.
    :param do_iterations: maximum number of times peaks are recursively split.
    :param min_area: Minimum area to do split. Smaller peaks are not split.
    :param n_top_channels: Number of top array channels. The function find_split_points(),
        implemented in each subclass defines the algorithm, which takes in a peak's waveform and
        returns the index to split the peak at, if a split point is found. Otherwise NO_MORE_SPLITS
        is returned and the peak is left as is.
    :param store_data_top: Boolean which indicates whether to store the top array waveform in the
        peak.
    :param store_data_start: Boolean which indicates whether to store the first samples of the
        waveform in the peak.

    """

    find_split_args_defaults: tuple

    def __call__(
        self,
        peaks,
        hits,
        records,
        rlinks,
        to_pe,
        data_type,
        do_iterations=1,
        min_area=0,
        n_top_channels=0,
        store_data_top=False,
        store_data_start=False,
        **kwargs,
    ):
        if not len(records) or not len(peaks) or not do_iterations:
            return peaks

        # Build the *args tuple for self.find_split_points from kwargs
        # since numba doesn't support **kwargs
        args_options = []
        for i, (k, value) in enumerate(self.find_split_args_defaults):
            if k in kwargs:
                value = kwargs[k]
            if k == "threshold":
                # The 'threshold' option is a user-specified function
                value = value(peaks)
            args_options.append(value)
        args_options = tuple(args_options)

        # Check for spurious options
        argnames = [k for k, _ in self.find_split_args_defaults]
        for k in kwargs:
            if k not in argnames:
                raise TypeError(f"Unknown argument {k} for {self.__class__}")

        is_split = np.zeros(len(peaks), dtype=bool)

        new_peaks = self._split_peaks(
            # Numba doesn't like self as argument, but it's ok with functions...
            split_finder=self.find_split_points,
            peaks=peaks,
            is_split=is_split,
            orig_dt=records[0]["dt"],
            min_area=min_area,
            args_options=tuple(args_options),
            result_dtype=peaks.dtype,
        )

        if is_split.sum() != 0:
            # Found new peaks: compute basic properties
            if data_type == "peaks":
                strax.sum_waveform(
                    new_peaks,
                    hits,
                    records,
                    rlinks,
                    to_pe,
                    n_top_channels=n_top_channels,
                    store_data_top=store_data_top,
                    store_data_start=store_data_start,
                )
                strax.compute_properties(new_peaks, n_top_channels=n_top_channels)
            elif data_type == "hitlets":
                # Add record fields here
                new_peaks = strax.sort_by_time(
                    new_peaks
                )  # Hitlets are not necessarily sorted after splitting
                new_peaks = strax.get_hitlets_data(new_peaks, records, to_pe)
            # ... and recurse (if needed)
            new_peaks = self(
                new_peaks,
                hits,
                records,
                rlinks,
                to_pe,
                data_type,
                do_iterations=do_iterations - 1,
                min_area=min_area,
                n_top_channels=n_top_channels,
                store_data_top=store_data_top,
                store_data_start=store_data_start,
                **kwargs,
            )
            if np.any(new_peaks["length"] == 0):
                raise ValueError("Want to add a new zero-length peak after splitting!")

            peaks = strax.sort_by_time(np.concatenate([peaks[~is_split], new_peaks]))

        return peaks

    # this function can not be cached due to some unknown reasons
    # maybe because the split_finder is a function and numba does not like it
    @staticmethod
    @strax.growing_result(dtype=strax.peak_dtype(), chunk_size=int(1e4))
    @numba.njit(nogil=True)
    def _split_peaks(
        split_finder,
        peaks,
        orig_dt,
        is_split,
        min_area,
        args_options,
        _result_buffer=None,
        result_dtype=None,
    ):
        """Loop over peaks, pass waveforms to algorithm, construct new peaks if and where a split
        occurs."""
        new_peaks = _result_buffer
        offset = 0

        for p_i, p in enumerate(peaks):
            if p["area"] < min_area:
                continue

            prev_split_i = 0
            w = p["data"][: p["length"]]
            for split_i, bonus_output in split_finder(w, p["dt"], p_i, *args_options):
                if split_i == NO_MORE_SPLITS:
                    p["max_goodness_of_split"] = bonus_output
                    # although the iteration will end anyway afterwards:
                    continue

                is_split[p_i] = True
                r = new_peaks[offset]
                r["time"] = p["time"] + prev_split_i * p["dt"]
                r["channel"] = p["channel"]
                # Set the dt to the original (lowest) dt first;
                # this may change when the sum waveform of the new peak
                # is computed
                r["dt"] = orig_dt
                r["length"] = (split_i - prev_split_i) * p["dt"] / orig_dt
                # Too lazy to compute these
                r["max_gap"] = -1
                r["max_diff"] = -1
                r["min_diff"] = -1
                r["first_channel"] = -1
                r["last_channel"] = -1
                if r["length"] <= 0:
                    print(p["data"])
                    print(prev_split_i, split_i)
                    raise ValueError("Attempt to create invalid peak!")

                offset += 1
                if offset == len(new_peaks):
                    yield offset
                    offset = 0

                prev_split_i = split_i

        yield offset

    @staticmethod
    def find_split_points(w, dt, peak_i, *args_options):
        """This function is overwritten by LocalMinimumSplitter or LocalMinimumSplitter bare
        PeakSplitter class is not implemented."""
        raise NotImplementedError


class LocalMinimumSplitter(PeakSplitter):
    """Split peaks at significant local minima.

    On either side of a split point, local maxima are required to be
     - larger than minimum + min_height, AND
     - larger than minimum * min_ratio
    This is related to topographical prominence for mountains.
    NB: Min_height is in pe/ns, NOT pe/bin!

    """

    find_split_args_defaults = (("min_height", 0), ("min_ratio", 0))

    @staticmethod
    @numba.njit(nogil=True)
    def find_split_points(w, dt, peak_i, min_height, min_ratio):
        """Yields indices of prominent local minima in w If there was at least one index, yields
        len(w)-1 at the end."""
        found_one = False
        last_max = -99999999999999.9
        min_since_max = 99999999999999.9
        min_since_max_i = 0

        for i, x in enumerate(w):
            if x < min_since_max:
                # New minimum since last max
                min_since_max = x
                min_since_max_i = i

            if min(last_max, x) > max(min_since_max + min_height, min_since_max * min_ratio):
                # Significant local minimum: tell caller,
                # reset both max and min finder
                yield min_since_max_i, 0.0
                found_one = True
                last_max = x
                min_since_max = 99999999999999.9
                min_since_max_i = i

            if x > last_max:
                # New max, reset minimum finder state
                # Notice this is AFTER the split check,
                # to accomodate very fast rising second peaks
                last_max = x
                min_since_max = 99999999999999.9
                min_since_max_i = i

        if found_one:
            yield len(w), 0.0
        yield NO_MORE_SPLITS, 0.0


class NaturalBreaksSplitter(PeakSplitter):
    """Split peaks according to (variations of) the natural breaks algorithm, i.e. such that the sum
    squared difference from the mean is minimized.

    Options:
     - threshold: threshold to accept a split in the goodness of split value:
       1 - (f(left) + f(right))/f(unsplit)
     - normalize: if True, f is the variance. Otherwise, it is the
       sum squared difference from the mean (i.e. unnormalized variance)
     - split_low: if True, multiply the goodness of split value by one minus
       the ratio between the waveform at the split point and the maximum in
       the waveform. This prevent splits at high density points.
     - filter_wing_width: if > 0, do a moving average filter (without shift)
       on the waveform before the split_low computation.
       The window will include the sample itself, plus filter_wing_width (or as
       close as we can get to it given the peaks sampling) on either side.

    """

    find_split_args_defaults = (
        ("threshold", None),  # will be a numpy array of len(peaks)
        ("normalize", False),
        ("split_low", False),
        ("filter_wing_width", 0),
    )

    @staticmethod
    @numba.njit(nogil=True)
    def find_split_points(w, dt, peak_i, threshold, normalize, split_low, filter_wing_width):
        gofs = natural_breaks_gof(
            w, dt, normalize=normalize, split_low=split_low, filter_wing_width=filter_wing_width
        )
        max_i = np.argmax(gofs)
        if gofs[max_i] > threshold[peak_i]:
            yield max_i, 0.0
            yield len(w) - 1, 0.0
        yield NO_MORE_SPLITS, gofs[max_i]


@export
@numba.njit(nogil=True, cache=True)
def natural_breaks_gof(w, dt, normalize=False, split_low=False, filter_wing_width=0):
    """Return natural breaks goodness of split/fit for the waveform w a sharp peak gives ~0, two
    widely separate peaks ~1."""
    left = sum_squared_deviations(w, normalize=normalize)
    right = sum_squared_deviations(w[::-1], normalize=normalize)[::-1]
    gof = 1 - (left + right) / left[-1]
    if split_low:
        # Adjust to prevent splits at high density points
        filter_n = filter_wing_width // dt - 1
        if filter_n > 0:
            filtered_w = symmetric_moving_average(w, filter_n)
        else:
            filtered_w = w
        gof *= 1 - filtered_w / filtered_w.max()
    return gof


@export
@numba.njit(nogil=True, cache=True)
def symmetric_moving_average(a, wing_width):
    """Return the moving average of a, over windows of length [2 * wing_width + 1] centered on each
    sample.

    (i.e. the window covers each sample itself, plus a 'wing' of width wing_width on either side)

    """
    if wing_width == 0:
        return a
    n = len(a)
    out = np.empty(n, dtype=a.dtype)
    asum = a[:wing_width].sum()
    count = wing_width
    for i in range(len(a)):
        # Index of the sample that just disappeared
        # from the window
        just_out = i - wing_width - 1
        if just_out > 0:
            count -= 1
            asum -= a[just_out]

        # Index of the sample that just appeared
        # in the window
        just_in = i + wing_width
        if just_in < n:
            count += 1
            asum += a[just_in]

        out[i] = asum / count
    return out


@numba.njit(nogil=True, cache=True)
def sum_squared_deviations(waveform, normalize=False):
    """Return left-to-right result of an online sum-intra-class variance computation on the
    waveform.

    :param normalize: If True, divide by the total area, i.e. produce ordinary variance.

    """
    mean = sum_weights = s = 0
    result = np.zeros(len(waveform))
    for i, w in enumerate(waveform):
        # Negative weights can lead to odd results, so clip waveform
        w = max(0, w)

        sum_weights += w
        if sum_weights == 0:
            continue

        mean_old = mean
        mean = mean_old + (w / sum_weights) * (i - mean_old)
        s += w * (i - mean_old) * (i - mean)
        result[i] = s
        if normalize:
            result[i] /= sum_weights

    return result
