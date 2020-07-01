import numpy as np
import numba

import strax
from strax import utils
from strax.dtypes import hit_dtype
export, __all__ = strax.exporter()

# TODO: Unify docsting style.

# ----------------------
# Hitlet building:
# ----------------------
@export
def concat_overlapping_hits(hits, extensions, pmt_channels):
    """
    Function which concatenates hits which may overlap after left and 
    right hit extension. Assumes that this are sorted correctly.
    
    Note: 
        This function only updates time, length and record_i of the hit.
        (record_i is set according to the first hit)
    
    :param hits: Hits in records.
    :param extensions: Tuple of the left and right hit extension.
    :param pmt_channels: Tuple of the detectors first and last PMT
    """
    # Getting channel map and compute the number of channels:
    first_channel, last_channel = pmt_channels
    nchannels = last_channel - first_channel + 1

    # Buffer for concat_overlapping_hits, if specified in 
    # _concat_overlapping_hits numba crashes.
    last_hit_in_channel = np.zeros(nchannels,
                                   dtype=(hit_dtype
                                          + [(('End time of the interval (ns since unix epoch)',
                                               'endtime'), np.int64)]))
    hits = _concat_overlapping_hits(hits, extensions, first_channel, last_hit_in_channel)
    return hits


@utils.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(hits,
                             extensions,
                             first_channel,
                             last_hit_in_channel,
                             _result_buffer=None):
    buffer = _result_buffer
    offset = 0

    le, re = extensions
    dt = hits['dt'][0]
    assert np.all(hits['dt'] == dt), 'All hits must have the same dt!'

    for h in hits:
        st = h['time'] - int(le * h['dt'])
        et = strax.endtime(h) + int(re * h['dt'])
        hc = h['channel']
        r_i = h['record_i']

        lhc = last_hit_in_channel[hc - first_channel]
        # Have not found any hit in this channel yet:
        if lhc['time'] == 0:
            lhc['time'] = st
            lhc['endtime'] = et
            lhc['channel'] = hc
            lhc['record_i'] = h['record_i']
            lhc['dt'] = dt

        # Checking if events overlap:
        else:
            if lhc['endtime'] >= st:
                # Yes, so we have to update only the end_time:
                lhc['endtime'] = et
            else:
                # No, this means we have to save the previous data and update lhc:
                res = buffer[offset]
                res['time'] = lhc['time']
                res['length'] = (lhc['endtime'] - lhc['time'])//lhc['dt']
                res['channel'] = lhc['channel']
                res['record_i'] = lhc['record_i']
                res['dt'] = lhc['dt']
                offset += 1
                if offset == len(buffer):
                    yield offset
                    offset = 0

                # Updating current last hit:
                lhc['time'] = st
                lhc['endtime'] = et
                lhc['channel'] = hc
                lhc['record_i'] = r_i

    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel['time'] != 0
    for lhc in last_hit_in_channel[mask]:
        res = buffer[offset]
        res['time'] = lhc['time']
        res['channel'] = lhc['channel']
        res['length'] = (lhc['endtime'] - lhc['time'])//lhc['dt']
        res['record_i'] = lhc['record_i']
        res['dt'] = lhc['dt']
        offset += 1
        if offset == len(buffer):
            yield offset
            offset = 0
    yield offset


@export
@numba.njit(nogil=True, cache=True)
def refresh_hit_to_hitlets(hits, hitlets):
    """
    Function which copies basic hit information into a new hitlet array.
    """
    for ind in range(len(hits)):
        h_new = hitlets[ind]
        h_old = hits[ind]

        h_new['time'] = h_old['time']
        h_new['length'] = h_old['length']
        h_new['channel'] = h_old['channel']
        h_new['area'] = h_old['area']
        h_new['record_i'] = h_old['record_i']
        h_new['dt'] = h_old['dt']


@export
@numba.njit(nogil=True, cache=True)
def get_hitlets_data(hitlets, records, to_pe):
    """
    Function which searches for every hitlet in a given chunk the 
    corresponding records data.
    
    :param hitlets: Hitlets found in a chunk of records.
    :param records: Records of the chunk.
    :param to_pe: Array with area conversion factors from adc/sample to 
        pe/sample
    
    Note:
        hitlets must have a "data" and "area" field.
    
    The function updates the hitlet fields time, length (if necessary 
    e.g. hit was extended in regions of now records) and area
    according to the found data.
    """
    # TODO: Add check for requirements of hitlets.

    rlink = strax.record_links(records)
    for h in hitlets:
        data, start_time = _get_hitlet_data(h, records, *rlink)
        h['length'] = len(data)
        h['data'][:len(data)] = data * to_pe[h['channel']]
        h['time'] = start_time
        h['area'] = np.sum(data * to_pe[h['channel']])


@numba.njit(nogil=True, cache=True)
def _get_hitlet_data(hitlet, records, prev_r, next_r):
    temp_data = np.zeros(hitlet['length'], dtype=np.float64)

    # Lets get the starting record and the corresponding data:
    r_i = hitlet['record_i']
    r = records[r_i]
    data, (p_start_i, p_end_i) = _get_thing_data(hitlet, r)
    temp_data[p_start_i:p_end_i] = data

    # We have to store the first and last index of the so far found data 
    data_start = p_start_i
    data_end = p_end_i

    if not (p_end_i - p_start_i == hitlet['length']):
        # We have not found the entire data yet....
        # Starting with data before our current record:
        trial_counter = 0
        prev_r_i = r_i
        while p_start_i:
            # We are still searching for data in a previous record.
            temp_prev_r_i = prev_r[prev_r_i]
            if temp_prev_r_i == -1:
                # There is no (more) previous record. So stop here and keep
                # last pre_r_i
                break
            prev_r_i = temp_prev_r_i

            # There is a previous record:
            r = records[prev_r_i]
            data, (p_start_i, end) = _get_thing_data(hitlet, r)
            if not end:
                raise ValueError('This is odd found previous record, but no'
                                 ' overlapping indices.')
            temp_data[p_start_i:data_start] = data
            data_start = p_start_i

            if trial_counter > 100:
                raise RuntimeError('Tried too hard. There are more than'
                                   '100 successive records. This is odd...')
            trial_counter += 1

        # Now we have to do the very same for records in the future:
        # Almost the same code as above sorry...
        trial_counter = 0
        next_r_i = r_i
        while hitlet['length'] - p_end_i:
            # We are still searching for data in a next record.
            temp_next_r_i = next_r[next_r_i]
            if temp_next_r_i == -1:
                # There is no (more) previous record. So stop here and keep
                # last next_r_i
                break
            next_r_i = temp_next_r_i
            # There is a next record:
            r = records[next_r_i]
            data, (start, p_end_i) = _get_thing_data(hitlet, r)
            if not start:
                raise ValueError('This is odd found the next record, but no'
                                 ' overlapping indicies.')
            temp_data[data_end:p_end_i] = data
            data_end = p_end_i

            if trial_counter > 100:
                raise RuntimeError('Tried too hard. There are more than'
                                   '100 successive records. This is odd...')
            trial_counter += 1

    # In some cases it might have happened that due to the left and right hit extension
    # we extended our hitlet into regions without any data so we have to chop
    # "time" according to the data we found....
    time = hitlet['time'] + data_start * hitlet['dt']
    temp_data = temp_data[data_start:data_end] + r['baseline'] % 1
    return temp_data, time


@numba.njit(nogil=True, cache=True)
def _get_thing_data(thing, container):
    """
    Function which returns data for some overlapping indices of a thing
    in a container. 
    
    Note:
        Thing must be of the interval dtype kind.
    """
    overlap_hit_i, overlap_record_i = strax.overlap_indices(thing['time']//thing['dt'],
                                                            thing['length'],
                                                            container['time']//container['dt'],
                                                            container['length'])
    data = container['data'][overlap_record_i[0]:overlap_record_i[1]]
    return data, overlap_hit_i

# ----------------------
# Hitlet splitting:
# ----------------------
@export
def update_new_hitlets(hitlets, records, next_ri, to_pe):
    """
    Function which computes the hitlet data area and record_i after
    splitting.

    :param hitlets: New hitlets received after splitting.
    :param records: Records of the chunk.
    :param next_ri: Index of next record for current record record_i.
    :param  to_pe: ADC to PE conversion factor array (of n_channels).
    """
    _update_record_i(hitlets, records, next_ri)
    get_hitlets_data(hitlets, records, to_pe)


@numba.njit(cache=True, nogil=True)
def _update_record_i(new_hitlets, records, next_ri):
    """
    Function which updates the record_i value of the new hitlets. 
    
    Notes:
        Assumes new_hitlets to be sorted in time.
    """
    for ind, hit in enumerate(new_hitlets):

        updated = False
        counter = 0
        current_ri = hit['record_i']
        while not updated:
            r = records[current_ri]
            # Hitlet must only partially be contained in record_i:
            time = hit['time']
            end_time = strax.endtime(hit)
            start_in = (r['time'] <= time) & (time < strax.endtime(r))
            end_in = (r['time'] < end_time) & (end_time <= strax.endtime(r))
            if start_in or end_in:
                hit['record_i'] = current_ri
                break
            else:
                last_ri = current_ri
                current_ri = next_ri[current_ri]
                counter += 1
                
            if current_ri == -1:
                print('Record:\n', r, '\nHit:\n', hit)
                raise ValueError('Was not able to find record_i')

            if counter > 100:
                print(ind, last_ri)
                raise RuntimeError('Tried too often to find correct record_i.')


# ----------------------
# Hitlet properties:
# ----------------------
@export
@numba.njit(cache=True, nogil=True)
def hitlet_properties(hitlets):
    """
    Computes additional lone hitlet properties.
    """
    for h in hitlets:

        dt = h['dt']
        data = h['data'][:h['length']]
        # Compute amplitude
        amp_ind = np.argmax(data)
        amp_time = int(amp_ind * dt)
        height = data[amp_ind]

        # Computing FWHM:
        left_edge, right_edge = get_fwxm(data, amp_ind, 0.5)
        left_edge = left_edge * dt + dt / 2
        right_edge = right_edge * dt - dt / 2
        width = right_edge - left_edge

        # Computing FWTM:
        left_edge_low, right_edge = get_fwxm(data, amp_ind, 0.1)
        left_edge_low = left_edge_low * dt + dt / 2
        right_edge = right_edge * dt - dt / 2
        width_low = right_edge - left_edge_low

        h['amplitude'] = height
        h['time_amplitude'] = amp_time
        h['fwhm'] = width
        h['left'] = left_edge
        h['low_left'] = left_edge_low
        h['fwtm'] = width_low


NO_FWXM = -42
@numba.njit(cache=True, nogil=True)
def get_fwxm(data, index_maximum, percentage=0.5):
    """
    Estimates the left and right edge of a specific height percentage.

    Args:
        data (np.array): Data of the pulse.
        index_maximum (ind): Position of the maximum.
        percentage (float): Level for which the width shall be computed.

    Notes:
        The function searches for the last sample below and above the
        specified height level on the left and right hand side of the
        maximum. When the samples are found the width is estimated based
        upon a linear interpolation between the respective samples. In
        case, that the samples cannot be found for either one of the
        sides the corresponding outer most bin edges are used: left 0;
        right last sample + 1.

    Returns:
        float: left edge [sample]
        float: right edge [sample]
    """
    max_val = data[index_maximum]
    max_val = max_val * percentage

    pre_max = data[:index_maximum]
    post_max = data[1 + index_maximum:]

    # First the left edge:
    lbi, lbs = _get_fwxm_boundary(pre_max, max_val)  # coming from the left
    if lbi == NO_FWXM:
        # We have not found any sample below:
        left_edge = 0.
    else:
        # We found a sample below so lets compute
        # the left edge:
        m = data[lbi + 1] - lbs  # divided by 1 sample
        left_edge = lbi + (max_val - lbs) / m

        # Now the right edge:
    rbi, rbs = _get_fwxm_boundary(post_max[::-1], max_val)  # coming from the right
    if rbi == NO_FWXM:
        right_edge = len(data)
    else:
        rbi = len(data) - rbi
        m = data[rbi - 2] - rbs
        right_edge = rbi - (max_val - data[rbi - 1]) / m

    return left_edge, right_edge


@numba.njit(cache=True, nogil=True)
def _get_fwxm_boundary(data, max_val):
    """
    Returns sample position and height for the last sample which amplitude is below
    the specified value
    """
    i = NO_FWXM
    s = NO_FWXM
    for ind, d in enumerate(data):
        if d < max_val:
            i = ind
            s = d
    return i, s