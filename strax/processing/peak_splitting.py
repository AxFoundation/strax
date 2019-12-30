import numpy as np
import numba
import strax

export, __all__ = strax.exporter()


@export
def split_peaks(peaks, records, to_pe, algorithm='local_minimum', **kwargs):
    """Return peaks split according to algorithm, with sum waveforms built.

    :param peaks: Original peaks. Sum waveform must have been built.
    :param records: Records from which peaks were built
    :param to_pe: ADC to PE conversion factor array (of n_channels)
    :param algorithm: 'local_minimum' or 'natural_breaks'.

    Any other options are passed to the algorithm.
    """
    splitter = dict(local_minimum=LocalMinimumSplitter,
                    natural_breaks=NaturalBreaksSplitter)[algorithm]()
    return splitter(peaks, records, to_pe, **kwargs)


class PeakSplitter:
    find_split_args_defaults: tuple

    def __call__(self, peaks, records, to_pe,
                 do_iterations=1, min_area=0, **kwargs):
        if not len(records) or not len(peaks):
            return peaks

        # Build the *args tuple for self.find_split_points from kwargs
        args_options = tuple([
            kwargs[k] if k in kwargs else default
            for k, default in self.find_split_args_defaults])

        # Check for spurious options
        argnames = [k for k, _ in self.find_split_args_defaults]
        for k in kwargs:
            if k not in argnames:
                raise TypeError(f"Unknown argument {k} for {self.__class__}")

        while do_iterations > 0:
            is_split = np.zeros(len(peaks), dtype=np.bool_)

            # Support for both algorithms is a bit awkward since we rely on numba,
            # so we can't use many of python's cool tricks to avoid duplication
            # (there surely is a way, but I'm too lazy to find one right now)
            new_peaks = self.split_peaks(
                # Numba doesn't like self as argument, but it's ok with functions...
                split_finder=self.find_split_points,
                peaks=peaks,
                is_split=is_split,
                orig_dt=records[0]['dt'],
                min_area=min_area,
                args_options=args_options,
                result_dtype=peaks.dtype)

            if is_split.sum() == 0:
                # Nothing was split -- end early
                break

            strax.sum_waveform(new_peaks, records, to_pe)
            peaks = strax.sort_by_time(np.concatenate([peaks[~is_split], new_peaks]))
            do_iterations -= 1

        return peaks

    @staticmethod
    @strax.growing_result(dtype=strax.peak_dtype(), chunk_size=int(1e4))
    @numba.jit(nopython=True, nogil=True, cache=True)
    def split_peaks(split_finder, peaks, orig_dt, is_split, min_area,
                    args_options,
                    _result_buffer=None, result_dtype=None):
        # TODO NEEDS TESTS!
        new_peaks = _result_buffer
        offset = 0

        for p_i, p in enumerate(peaks):
            if p['area'] < min_area:
                continue

            prev_split_i = 0
            w = p['data'][:p['length']]

            for split_i in split_finder(w, p['dt'], *args_options):
                is_split[p_i] = True
                r = new_peaks[offset]
                r['time'] = p['time'] + prev_split_i * p['dt']
                r['channel'] = p['channel']
                # Set the dt to the original (lowest) dt first;
                # this may change when the sum waveform of the new peak is computed
                r['dt'] = orig_dt
                r['length'] = (split_i - prev_split_i) * p['dt'] / orig_dt

                if r['length'] <= 0:
                    print(p['data'])
                    print(prev_split_i, split_i)
                    raise ValueError("Attempt to create invalid peak!")

                offset += 1
                if offset == len(new_peaks):
                    yield offset
                    offset = 0

                prev_split_i = split_i

        yield offset

    @staticmethod
    def find_split_points(w, *args_options):
        raise NotImplementedError


class LocalMinimumSplitter(PeakSplitter):
    """Split peaks at significant local minima.

    On either side of a split point, local maxima are required to be
     - larger than minimum + min_height, AND
     - larger than minimum * min_ratio
    This is related to topographical prominence for mountains.
    NB: Min_height is in pe/ns, NOT pe/bin!
    """
    find_split_args_defaults = (
        ('min_height', 0),
        ('min_ratio', 0),
    )

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_split_points(w, dt, min_height, min_ratio):
        """"Yield indices of prominent local minima in w
        If there was at least one index, yields len(w)-1 at the end
        """
        found_one = False
        last_max = -99999999999999.9
        min_since_max = 99999999999999.9
        min_since_max_i = 0

        for i, x in enumerate(w):
            if x < min_since_max:
                # New minimum since last max
                min_since_max = x
                min_since_max_i = i

            if min(last_max, x) > max(min_since_max + min_height,
                                      min_since_max * min_ratio):
                # Significant local minimum: tell caller,
                # reset both max and min finder
                yield min_since_max_i
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
            yield len(w)


class NaturalBreaksSplitter(PeakSplitter):
    """Split peaks according to (variations of) the natural breaks algorithm,
    i.e. such that the sum squared difference from the mean is minimized.

    Options:
     - threshold: threshold to accept a split in the goodness of split value:
       1 - (f(left) + f(right))/f(unsplit)
     - normalize: if True, f is the variance. Otherwise, it is the
       sum squared difference from the mean (i.e. unnormalized variance)
     - split_low: if True, multiply the goodness of split value by the ratio
       between the waveform at the split point and the maximum in the waveform.
       This prevent splits at high density points.
     - filter_wing_width: if > 0, do a moving average filter (without shift) on the
       waveform before the split_low computation.
       The window will include the sample itself, plus filter_wing_width (or as
       close as we can get to it given the peaks sampling) on either side.
    """
    find_split_args_defaults = (
        ('threshold', 0.4),
        ('normalize', False),
        ('split_low', False),
        ('filter_wing_width', 0))

    @staticmethod
    @numba.njit(nogil=True, cache=True)
    def find_split_points(w, dt, threshold, normalize, split_low, filter_wing_width):
        gofs = natural_breaks_gof(w, dt,
                                  normalize=normalize,
                                  split_low=split_low,
                                  filter_wing_width=filter_wing_width)
        i = np.argmax(gofs)
        if gofs[i] > threshold:
            yield i
            yield len(w) - 1


@export
@numba.njit(nogil=True, cache=True)
def natural_breaks_gof(w, dt, normalize=False, split_low=False, filter_wing_width=0):
    """Return natural breaks goodness of split/fit for the waveform w
    a sharp peak gives ~0, two widely separate peaks ~1.
    """
    left = sum_squared_deviations(w, normalize=normalize)
    right = sum_squared_deviations(w[::-1], normalize=normalize)[::-1]
    gof = 1 - (left + right) / left[-1]
    if split_low:
        # Adjust to prevent splits at high density points
        filter_width = filter_wing_width
        filter_n = filter_width // dt - 1
        if filter_n > 0:
            filtered_w = symmetric_moving_average(w, filter_n)
        else:
            filtered_w = w
        gof *= 1 - filtered_w / filtered_w.max()
    return gof


@export
@numba.njit(nogil=True, cache=True)
def symmetric_moving_average(a, wing_width):
    """Return the moving average of a, over windows
    of length [2 * wing_width + 1] centered on each sample.

    (i.e. the window covers each sample itself, plus a 'wing' of width
     wing_width on either side)
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
    """Return left-to-right result of an online
    sum-intra-class variance computation on the waveform.

    :param normalize: If True, divide by the total area,
    i.e. produce ordinary variance.
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
