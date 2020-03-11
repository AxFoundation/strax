from hypothesis import given, settings

from strax.testutils import *


# TODO: test with multiple fake pulses and dt != 1
@settings(deadline=None)
@given(single_fake_pulse)
def test_cut_outside_hits(records):
    hits = strax.find_hits(records, min_amplitude=1)

    # Set all record waveforms to 1 (still and 0 out of bounds)
    for r in records:
        r['data'] = 0
        r['data'][:r['length']] = 1
        assert np.all(np.in1d(r['data'], [0, 1]))

    left_extension = 2
    right_extension = 3

    records_out = strax.cut_outside_hits(
        records,
        hits,
        left_extension=left_extension,
        right_extension=right_extension)

    assert len(records_out) == len(records)
    if len(records) == 0:
        return

    # All fields except data are unchanged
    for x in records.dtype.names:
        if x == 'data':
            continue
        if x == 'reduction_level':
            np.testing.assert_array_equal(
                records_out[x],
                np.ones(len(records), dtype=np.int16)
                * strax.ReductionLevel.HITS_ONLY)
        else:
            np.testing.assert_array_equal(records_out[x], records[x],
                                          err_msg=f"Field {x} mangled!")

    records = records_out

    # Super-laborious dumb check
    for r in records:
        for i, w in enumerate(r['data'][:r['length']]):
            t = r['time'] + i * r['dt']
            for h in hits:
                if (h['time'] - left_extension
                        <= t <
                        strax.endtime(h) + right_extension):
                    assert w == 1, f"Position {i} should be preserved"
                    break
            else:
                assert w == 0, f"Position {i} should be cut"
