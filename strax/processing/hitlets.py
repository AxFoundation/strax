import numpy as np
import numba

import strax
export, __all__ = strax.exporter()

# Hardcoded numbers:
TRIAL_COUNTER_NEIGHBORING_RECORDS = 100  # Trial counter when looking for hitlet data.
NO_FWXM = -42  # Value in case FWXM cannot be found.

# ----------------------
# Hitlet building:
# ----------------------
@export
def concat_overlapping_hits(hits, extensions, pmt_channels, start, end):
    """
    Function which concatenates hits which may overlap after left and 
    right hit extension. Assumes that hits are sorted correctly.

    Note:
        This function only updates time, length and record_i of the hit.
        (record_i is set according to the first hit)

    :param hits: Hits in records.
    :param extensions: Tuple of the left and right hit extension.
    :param pmt_channels: Tuple of the detectors first and last PMT
    :param start: Startime of the chunk
    :param end: Endtime of the chunk

    :returns:
        array with concataneted hits.
    """
    # Getting channel map and compute the number of channels:
    first_channel, last_channel = pmt_channels
    nchannels = last_channel - first_channel + 1

    # Buffer for concat_overlapping_hits, if specified in 
    # _concat_overlapping_hits numba crashes.
    last_hit_in_channel = np.zeros(nchannels,
                                   dtype=(strax.hit_dtype
                                          + [(('End time of the interval (ns since unix epoch)',
                                               'endtime'), np.int64)]))

    if len(hits):
        hits = _concat_overlapping_hits(hits, extensions, first_channel, last_hit_in_channel, start, end)
    return hits


@strax.utils.growing_result(strax.hit_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(hits,
                             extensions,
                             first_channel,
                             last_hit_in_channel,
                             start=0,
                             end=float('inf'),
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
            lhc['time'] = max(st, start)
            lhc['endtime'] = min(et, end)
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
                res['length'] = (lhc['endtime'] - lhc['time']) // lhc['dt']
                res['channel'] = lhc['channel']
                res['record_i'] = lhc['record_i']
                res['dt'] = lhc['dt']
                
                # Updating current last hit:
                lhc['time'] = st
                lhc['endtime'] = et
                lhc['channel'] = hc
                lhc['record_i'] = r_i
                
                offset += 1
                if offset == len(buffer):
                    yield offset
                    offset = 0

    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel['time'] != 0
    for lhc in last_hit_in_channel[mask]:
        res = buffer[offset]
        res['time'] = lhc['time']
        res['channel'] = lhc['channel']
        res['length'] = (lhc['endtime'] - lhc['time']) // lhc['dt']
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
    nhits = len(hits)
    for ind in range(nhits):
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

    rlink = strax.record_links(records)
    for h in hitlets:
        data, start_time = get_single_hitlet_data(h, records, *rlink)
        h['length'] = len(data)
        h['data'][:len(data)] = data * to_pe[h['channel']]
        h['time'] = start_time
        h['area'] = np.sum(data * to_pe[h['channel']])


@export
@numba.njit(nogil=True, cache=True)
def get_single_hitlet_data(hitlet, records, prev_r, next_r):
    """
    Function which gets the data of a single hit or hitlet. The data is
    returned according to the objects time and length (LE/RE is not
    included in case of a hit.).

    In case the hit or hitlet is extended into non-recorded regions
    the data gets chopped.


    :param hitlet: Hits or hitlets.
    :param records: Records
    :param prev_r: Index of the previous record seen from the current
        record. (Return of strax.record_links)
    :param next_r: Index of the next record seen by the current record.
        (Return of strax.record_links)
    :return:
        np.ndarray: Samples of the hitlet [ADC]
        int: Start time of the hitlet. (In case data gets chopped on the
            left)
    """
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
                # If end is zero this means we have not found any
                # overlap which should not have happened. In case of an
                # overlap start and end should reflect the start and end
                # sample of the hitlet for which we found data.
                print('Data found for this record:', data,
                      'Start index', p_start_i,
                      'End index:', end)
                raise ValueError('This is odd found previous record, but no'
                                 ' overlapping indices.')
            temp_data[p_start_i:data_start] = data
            data_start = p_start_i

            if trial_counter > TRIAL_COUNTER_NEIGHBORING_RECORDS:
                raise RuntimeError('Tried too hard. There are more than'
                                   '100 successive records. This is odd...')
            trial_counter += 1

        # Now we have to do the very same for records in the future:
        # Almost the same code as above can I change this?
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
                # If start is zero this means we have not found any
                # overlap which should not have happened. In case of an
                # overlap start and end should reflect the start and end
                # sample of the hitlet for which we found data.
                print('Data found for this record:', data,
                      'Start index', start,
                      'End index:', p_end_i)
                raise ValueError('This is odd found the next record, but no'
                                 ' overlapping indicies.')
            temp_data[data_end:p_end_i] = data
            data_end = p_end_i

            if trial_counter > TRIAL_COUNTER_NEIGHBORING_RECORDS:
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

            if counter > TRIAL_COUNTER_NEIGHBORING_RECORDS:
                print(ind, last_ri)
                raise RuntimeError('Tried too often to find correct record_i.')


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
    if np.all(data > max_val) or np.all(data == 0):
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
        left_edge = lbi + (max_val - lbs) / m + 0.5
    else:
        # There is no data before the maximum:
        left_edge = 0

    if len(post_max) and np.any(post_max <= max_val):
        # Now the right edge:
        rbi, rbs = _get_fwxm_boundary(post_max, max_val)  # Starting after maximum and go right
        rbi += 1 + index_maximum  # sample to the right plus start
        m = data[rbi - 1] - rbs
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
    inter, amps = strax.highest_density_region(data, fractions_desired, _buffer_size=_buffer_size)

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
