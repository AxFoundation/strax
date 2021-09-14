import numpy as np
import numba

import strax
from strax.processing.general import _touching_windows

export, __all__ = strax.exporter()

# Hardcoded numbers:
NO_FWXM = -42  # Value in case FWXM cannot be found.


@export
def create_hitlets_from_hits(hits,
                             save_outside_hits,
                             channel_range,
                             chunk_start=0,
                             chunk_end=np.inf,):
    """
    Function which creates hitlets from a bunch of hits.

    :param hits: Hits found in records.
    :param save_outside_hits: Tuple with left and right hit extension.
    :param channel_range: Detectors change range from channel map.
    :param chunk_start: (optional) start time of a chunk. Ensures that
        no hitlet is earlier than this timestamp.
    :param chunk_end: (optional) end time of a chunk. Ensures that
        no hitlet ends later than this timestamp.

    :return: Hitlets with temporary fields (data, max_goodness_of_split...)
    """
    # Merge concatenate overlapping  within a channel. This is important
    # in case hits were split by record boundaries. In case we
    # accidentally concatenate two PMT signals we split them later again.
    hits = strax.concat_overlapping_hits(hits,
                                         save_outside_hits,
                                         channel_range,
                                         chunk_start,
                                         chunk_end, )
    hits = strax.sort_by_time(hits)

    hitlets = np.zeros(len(hits), strax.hitlet_dtype())
    strax.copy_to_buffer(hits, hitlets, '_refresh_hit_to_hitlets')
    return hitlets


@export
def concat_overlapping_hits(hits, extensions, pmt_channels, start, end):
    """
    Function which concatenates hits which may overlap after left and 
    right hit extension. Assumes that hits are sorted correctly.

    Note:
        This function only updates time, and length of the hit.

    :param hits: Hits in records.
    :param extensions: Tuple of the left and right hit extension.
    :param pmt_channels: Tuple of the detectors first and last PMT
    :param start: Startime of the chunk
    :param end: Endtime of the chunk

    :returns:
        array with concataneted hits.
    """
    first_channel, last_channel = pmt_channels
    nchannels = last_channel - first_channel + 1

    # Buffer for concat_overlapping_hits, if specified in 
    # _concat_overlapping_hits numba crashes.
    last_hit_in_channel = np.zeros(nchannels,
                                   dtype=(strax.hit_dtype
                                          + [(('End time of the interval (ns since unix epoch)',
                                               'endtime'), np.int64)]))

    if len(hits):
        hits = _concat_overlapping_hits(
            hits, extensions, first_channel, last_hit_in_channel, start, end)
    return hits


@strax.utils.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(hits,
                             extensions,
                             first_channel,
                             last_hit_in_channel_buffer,
                             chunk_start=0,
                             chunk_end=float('inf'),
                             _result_buffer=None):
    buffer = _result_buffer
    res_offset = 0

    left_extension, right_extension = extensions
    dt = hits['dt'][0]
    assert np.all(hits['dt'] == dt), 'All hits must have the same dt!'

    for hit in hits:
        time_with_le = hit['time'] - int(left_extension * hit['dt'])
        endtime_with_re = strax.endtime(hit) + int(right_extension * hit['dt'])
        hit_channel = hit['channel']

        last_hit_in_channel = last_hit_in_channel_buffer[hit_channel - first_channel]

        found_no_hit_for_channel_yet = last_hit_in_channel['time'] == 0
        if found_no_hit_for_channel_yet:
            last_hit_in_channel['time'] = max(time_with_le, chunk_start)
            last_hit_in_channel['endtime'] = min(endtime_with_re, chunk_end)
            last_hit_in_channel['channel'] = hit_channel
            last_hit_in_channel['dt'] = dt
        else:
            hits_overlap_in_channel = last_hit_in_channel['endtime'] >= time_with_le
            if hits_overlap_in_channel:
                last_hit_in_channel['endtime'] = endtime_with_re
            else:
                # No, this means we have to save the previous data and update lhc:
                res = buffer[res_offset]
                res['time'] = last_hit_in_channel['time']
                hitlet_length = (last_hit_in_channel['endtime'] - last_hit_in_channel['time'])
                hitlet_length //= last_hit_in_channel['dt']
                res['length'] = hitlet_length
                res['channel'] = last_hit_in_channel['channel']
                res['dt'] = last_hit_in_channel['dt']
                
                # Updating current last hit:
                last_hit_in_channel['time'] = time_with_le
                last_hit_in_channel['endtime'] = endtime_with_re
                last_hit_in_channel['channel'] = hit_channel
                
                res_offset += 1
                if res_offset == len(buffer):
                    yield res_offset
                    res_offset = 0

    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel_buffer['time'] != 0
    for last_hit_in_channel in last_hit_in_channel_buffer[mask]:
        res = buffer[res_offset]
        res['time'] = last_hit_in_channel['time']
        res['channel'] = last_hit_in_channel['channel']
        hitlet_length = (last_hit_in_channel['endtime'] - last_hit_in_channel['time'])
        hitlet_length //= last_hit_in_channel['dt']
        res['length'] = hitlet_length
        res['dt'] = last_hit_in_channel['dt']
        res_offset += 1
        if res_offset == len(buffer):
            yield res_offset
            res_offset = 0
    yield res_offset


@export
def get_hitlets_data(hitlets, records, to_pe, min_hitlet_sample=200):
    """
    Function which searches for every hitlet in a given chunk the 
    corresponding records data. Additionally compute the total area of
    the signal.

    :param hitlets: Hitlets found in a chunk of records.
    :param records: Records of the chunk.
    :param to_pe: Array with area conversion factors from adc/sample to
        pe/sample. Please make sure that to_pe has the correct shape.
        The array index should match the channel number.
    :param min_hitlet_sample: minimal length of the hitlet data field.
        prevents numba compiling from running into race conditions.
    :returns: Hitlets including data stored in the "data" field
        (if it did not exists before it will be added.)
    """
    if len(hitlets) == 0:
        return np.zeros(0, dtype=strax.hitlet_with_data_dtype(min_hitlet_sample))

    if len(hitlets) > 0 and len(records) == 0:
        raise ValueError('Cannot get data for hitlets if records are empty!')

    # Numba will not raise any exceptions if to_pe is too short, leading
    # to strange bugs.
    to_pe_has_wrong_shape = len(to_pe) < hitlets['channel'].max()
    if to_pe_has_wrong_shape:
        raise ValueError('"to_pe" has a wrong shape. Array index must'
                         ' match channel numbers.')

    hitelts_is_single_row = isinstance(hitlets, np.void)
    if hitelts_is_single_row:
        # A structured array becomes void type if a single row is called,
        # e.g. hitlets[0] which does not work in numba while, hitlets[:1]
        # does. So we have to convert the row into the correct format first.
        hitlets = np.array([hitlets])

    data_field_in_hitlets = 'data' in hitlets.dtype.names
    if data_field_in_hitlets:
        data_is_not_empty = np.any(hitlets['data'] != 0)
        if data_is_not_empty:
            raise ValueError('The data field of hitlets must be empty!')

        data_field_not_long_enough = len(hitlets[0]['data']) < hitlets['length'].max()
        if data_field_not_long_enough:
            raise ValueError('The data field must be as large as the longest hitlet in our data.')

        hitlets_with_data_field = hitlets
    else:
        n_samples = max(min_hitlet_sample, hitlets['length'].max())
        hitlets_with_data_field = np.zeros(len(hitlets), strax.hitlet_with_data_dtype(n_samples))
        strax.copy_to_buffer(hitlets,
                             hitlets_with_data_field,
                             '_copy_hitlets_to_hitlets_width_data')

    _get_hitlets_data(hitlets_with_data_field, records, to_pe)
    return hitlets_with_data_field


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_hitlets_data(hitlets, records, to_pe):
    rranges = _touching_windows(records['time'],
                                strax.endtime(records),
                                hitlets['time'],
                                strax.endtime(hitlets))

    for i, h in enumerate(hitlets):
        recorded_samples_offset = 0
        n_recorded_samples = 0
        is_first_record = True
        for ind, r_ind in enumerate(range(rranges[i][0], rranges[i][1])):
            r = records[r_ind]
            if r['channel'] != h['channel']:
                continue

            (r_start, r_end), (h_start, h_end) = strax.overlap_indices(
                r['time'] // r['dt'],
                r['length'],
                h['time'] // h['dt'],
                h['length'])

            if is_first_record:
                # We need recorded_samples_offset because hits may extend beyond the boundaries of our recorded data.
                # As the data is not defined in those regions we have to chop and realign our data. See the following
                # Example: (fragment 0, 1) [2, 2, 2, 2] [2, 2, 2] with a hitfinder threshold of 1 and left/right
                # extension of 3. In the first fragment our hitlet would range from 3 to 8 in the second from 8
                # to 11. Hence we have to subtract from every h_start and h_end the offset of 3 to realign our data.
                # Time and length of the hitlet are updated accordingly.
                is_first_record = False
                recorded_samples_offset = h_start
            h_start -= recorded_samples_offset
            h_end -= recorded_samples_offset

            h['data'][h_start: h_end] += r['data'][r_start: r_end] + r['baseline'] % 1
            n_recorded_samples += r_end - r_start

        # Chop time and length in case hit extends into non-recorded regions.
        h['time'] += int(recorded_samples_offset * h['dt'])
        h['length'] = n_recorded_samples

        h['data'][:] = h['data'][:] * to_pe[h['channel']]
        h['area'] = np.sum(h['data'])


# ----------------------
# Hitlet properties:
# ----------------------
@export
@numba.njit(cache=True, nogil=True)
def hitlet_properties(hitlets):
    """
    Computes additional hitlet properties such as amplitude, FHWM, etc.
    """
    for ind, h in enumerate(hitlets):
        dt = h['dt']
        data = h['data'][:h['length']]

        if not np.any(data):
            continue

        # Compute amplitude
        amp_ind = np.argmax(data)
        amp_time = int(amp_ind * dt)
        height = data[amp_ind]

        h['amplitude'] = height
        h['time_amplitude'] = amp_time

        # Computing FWHM:
        left_edge, right_edge = get_fwxm(h, 0.5)
        width = right_edge - left_edge

        # Computing FWTM:
        left_edge_low, right_edge = get_fwxm(h, 0.1)
        width_low = right_edge - left_edge_low

        h['fwhm'] = width
        h['left'] = left_edge
        h['low_left'] = left_edge_low
        h['fwtm'] = width_low

        # Compute area deciles & width:
        if not h['area'] == 0:
            # Due to noise total area can sum up to zero
            res = np.zeros(4, dtype=np.float32)
            deciles = np.array([0.1, 0.25, 0.75, 0.9])
            strax.compute_index_of_fraction(h, deciles, res)
            res *= h['dt']
            
            h['left_area'] = res[1]
            h['low_left_area'] = res[0]
            h['range_50p_area'] = res[2]-res[1]
            h['range_80p_area'] = res[3]-res[0]
            
        # Compute width based on HDR:
        resh = highest_density_region_width(data, 
                                            fractions_desired=np.array([0.5, 0.8]),
                                            dt=h['dt'],
                                            fractionl_edges=True,
                                            )

        h['left_hdr'] = resh[0,0]
        h['low_left_hdr'] = resh[1,0]
        h['range_hdr_50p_area'] = resh[0,1]-resh[0,0]
        h['range_hdr_80p_area'] = resh[1,1]-resh[1,0]


@export
@numba.njit(cache=True, nogil=True)
def get_fwxm(hitlet, fraction=0.5):
    """
    Estimates the left and right edge of a specific height percentage.

    :param hitlet: Single hitlet
    :param fraction: Level for which the width shall be computed.
    :returns: Two floats, left edge and right edge in ns

    Notes:
        The function searches for the last sample below and above the
        specified height level on the left and right hand side of the
        maximum. When the samples are found the width is estimated based
        upon a linear interpolation between the respective samples. In
        case, that the samples cannot be found for either one of the
        sides the corresponding outer most bin edges are used: left 0;
        right last sample + 1.
    """
    data = hitlet['data'][:hitlet['length']]

    index_maximum = np.argmax(data)
    max_val = data[index_maximum] * fraction
    if np.all(data > max_val) or np.all(data <= 0):
        # In case all samples are larger, FWXM is not definition.
        return np.nan, np.nan

    pre_max = data[:index_maximum]  # Does not include maximum
    post_max = data[1 + index_maximum:]  # same

    if len(pre_max) and np.any(pre_max <= max_val):
        # First the left edge:

        lbi, lbs = _get_fwxm_boundary(pre_max[::-1], max_val)  # Reversing data starting at sample
        # before maximum and go left
        lbi = (index_maximum - 1) - lbi  # start sample minus samples we went to the left
        m = data[lbi + 1] - lbs  # divided by 1 sample
        if m == 0:
            return np.nan, np.nan
        left_edge = lbi + (max_val - lbs) / m + 0.5
    else:
        # There is no data before the maximum:
        left_edge = 0

    if len(post_max) and np.any(post_max <= max_val):
        # Now the right edge:
        rbi, rbs = _get_fwxm_boundary(post_max, max_val)  # Starting after maximum and go right
        rbi += 1 + index_maximum  # sample to the right plus start
        m = data[rbi - 1] - rbs
        if m == 0:
            return np.nan, np.nan
        right_edge = rbi - (max_val - rbs) / m + 0.5
    else:
        right_edge = len(data)

    left_edge = left_edge * hitlet['dt']
    right_edge = right_edge * hitlet['dt']
    return left_edge, right_edge


@numba.njit(cache=True, nogil=True)
def _get_fwxm_boundary(data, max_val):
    """
    Returns sample position and height for the last sample which
    amplitude is below the specified value.

    If no sample can be found returns position and value of last sample
    seen.

    Note:
        For FWHM we assume that we start at the maximum.
    """
    ind = None
    s = None
    for i, d in enumerate(data):
        if d <= max_val:
            ind = i
            s = d
            return ind, s
    return len(data)-1, data[-1]


@export
def conditional_entropy(hitlets, template='flat', square_data=False):
    """
    Function which estimates the conditional entropy based on the
    specified template.

    In order to compute the conditional entropy each hitlet will be
    aligned such that its maximum falls into the same sample as for the
    template. If the maximum is ambiguous the first maximum is taken.

    :param hitlets: Hitlets for which the entropy shall be computed.
        Can be any data_kind which offers the fields data and length.
    :param template: Template to compare the data with. Can be either
        specified as "flat" to use a flat distribution or as a numpy
        array containing any normalized template.
    :param square_data: If true data will be squared and normalized
        before estimating the entropy. Otherwise the data will only be
        normalized.
    :returns: Array containing the entropy values for each hitlet.

    Note:
        The template has to be normalized such that its total area is 1.
        Independently of the specified options, only samples for which
        the content is greater zero are used to compute the entropy.

        In case of the non-squared case negative samples are omitted in
        the calculation.
    """
    if not isinstance(template, np.ndarray) and template != 'flat':
        raise ValueError('Template input not understood. Must be either a numpy array,\n',
                         'or "flat".')

    if 'data' not in hitlets.dtype.names:
        raise ValueError('"hitlets" must have a field "data".')

    if isinstance(template, str) and template == 'flat':
        template = np.empty(0, dtype=np.float32)
        flat = True
    else:
        flat = False
    res = _conditional_entropy(hitlets, template, flat=flat,
                               square_data=square_data)
    return res


@numba.njit(cache=True, nogil=True)
def _conditional_entropy(hitlets, template, flat=False, square_data=False):
    res = np.zeros(len(hitlets), dtype=np.float32)
    for ind, h in enumerate(hitlets):
        # h['data'][:] generates just a view and not a copy of the data
        # Since the data of hitlets should not be modified we have
        # to use a copy instead.... See also https://stackoverflow.com/questions/4370745/view-onto-a-numpy-array
        hitlet = np.copy(h['data'][:h['length']])

        # Squaring and normalizing:
        if square_data:
            hitlet[:] = hitlet * hitlet
        if np.sum(hitlet):
            hitlet[:] = hitlet / np.sum(hitlet)
        else:
            # If there is no area we cannot normalize
            res[ind] = np.nan
            continue

        if flat:
            # Take out values which are smaller euqal zero since:
            # lim x_i --> 0, x_i * log(x_i) --> 0
            # and log not defined for negative values.
            m = hitlet > 0
            hitlet = hitlet[m]
            len_hitlet = len(hitlet)

            template = np.ones(len_hitlet, dtype=np.float32)
            template = template / np.sum(template)

            e = - np.sum(hitlet * np.log(hitlet / template))
        else:
            # In case of a template we take out zeros and negative values
            # once we populated the buffer. Otherwise we miss-align
            # template and buffer.

            # create a buffers to align data and template:
            len_hitlet = len(hitlet)
            len_template = len(template)
            length = np.max(np.array([len_hitlet, len_template]))
            length = length * 2 + 1
            buffer = np.zeros((2, length), dtype=np.float32)

            # align data and template and compute entropy:
            si = length // 2 - np.argmax(hitlet)
            ei = si + len(hitlet)
            buffer[0, si:ei] = hitlet[:]

            si = length // 2 - np.argmax(template)
            ei = si + len(template)
            buffer[1, si:ei] = template[:]

            # Remove zeros from buffers:
            m_hit = (buffer[0] > 0)
            m_temp = (buffer[1] > 0)
            m = m_hit & m_temp
            e = - np.sum(buffer[0][m] * np.log(buffer[0][m] / buffer[1][m]))
        res[ind] = e
    return res


@numba.njit
def highest_density_region_width(data,
                                  fractions_desired,
                                  dt=1,
                                  fractionl_edges=False,
                                  _buffer_size=100):
    """
    Function which computes the left and right edge based on the outer
    most sample for the highest density region of a signal.

    Defines a 100% fraction as the sum over all positive samples in a
    waveform.

    :param data: Data of a signal, e.g. hitlet or peak including zero length
        encoding.
    :param fractions_desired: Area fractions for which the highest
        density region should be computed.
    :param dt: Sample length in ns.
    :param fractionl_edges: If true computes width as fractional time
        depending on the covered area between the current and next
        sample.
    :param _buffer_size: Maximal number of allowed intervals.
    """
    res = np.zeros((len(fractions_desired), 2), dtype=np.float32)
    data = np.maximum(data, 0)
    
    if np.all(data == 0):
        res[:] = np.nan
        return res
    else:
        inter, amps = strax.highest_density_region(data,
                                                   fractions_desired, _buffer_size=_buffer_size)

    for f_ind, (i, a) in enumerate(zip(inter, amps)):
        if not fractionl_edges:
            res[f_ind, 0] = i[0, 0] * dt
            res[f_ind, 1] = i[1, np.argmax(i[1, :])] * dt
        else:
            left = i[0, 0]
            right = i[1, np.argmax(i[1, :])] - 1  # since value corresponds to outer edge

            # Get amplitudes of outer most samples
            # and amplitudes of adjacent samples (if any)
            left_amp = data[left]
            right_amp = data[right]

            next_left_amp = 0
            if (left - 1) >= 0:
                next_left_amp = data[left - 1]
            next_right_amp = 0
            if (right + 1) < len(data):
                next_right_amp = data[right + 1]

            # Compute fractions and new left and right edges:
            fl = (left_amp - a) / (left_amp - next_left_amp)
            fr = (right_amp - a) / (right_amp - next_right_amp)

            res[f_ind, 0] = (left + 0.5 - fl) * dt
            res[f_ind, 1] = (right + 0.5 + fr) * dt

    return res
