"""
Convert pax .zip files to flat records format

#implement main logic as pax output plugin?
#Hard, would have to append to mmap...
"""
import numpy as np

from strax.dtypes import record_dtype
from strax.utils import records_needed


def pax_to_records(input_filename, samples_per_record=110):
    """Return pulse records array from pax zip input_filename"""
    from pax import core   # Pax is not a dependency
    mypax = core.Processor('XENON1T', config_dict=dict(
            pax=dict(
                look_for_config_in_runs_db=False,
                plugin_group_names=['input'],
                encoder_plugin=None,
                input_name=input_filename),
            # Fast startup: skip loading big maps
            WaveformSimulator=dict(
                s1_light_yield_map='placeholder_map.json',
                s2_light_yield_map='placeholder_map.json',
                s1_patterns_file=None,
                s2_patterns_file=None)))

    def get_events():
        for e in mypax.get_events():
            yield mypax.process_event(e)

    # We loop over the events twice for convenience
    # Yeah, this is probably not optimal
    pulse_lengths = np.array([p.length
                              for e in get_events()
                              for p in e.pulses])

    n_records = records_needed(pulse_lengths, samples_per_record).sum()
    records = np.zeros(n_records, dtype=record_dtype(samples_per_record))

    i = 0  # Record offset in data
    for event in get_events():
        for p in event.pulses:

            n_records = records_needed(p.length, samples_per_record)
            for rec_i in range(n_records):

                records[i]['time'] = event.start_time \
                                     + p.left * 10 \
                                     + rec_i * samples_per_record * 10
                records[i]['channel'] = p.channel
                records[i]['total_length'] = p.length
                records[i]['record_i'] = rec_i

                # How much are we storing in this record?
                if rec_i != n_records - 1:
                    # There's more chunks coming, so we store a full chunk
                    n_store = samples_per_record
                else:
                    # Just enough to store the rest of the data
                    n_store = p.length % samples_per_record

                offset = rec_i * samples_per_record
                records[i]['data'][:n_store] = \
                    p.raw_data[offset:offset + n_store]
                i += 1

    mypax.shutdown()
    return records
