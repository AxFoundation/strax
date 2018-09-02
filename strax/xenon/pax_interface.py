"""Convert pax .zip files to flat records format
"""
import numpy as np
import os
import glob

import strax
export, __all__ = strax.exporter()


@export
def pax_to_records(input_filename,
                   samples_per_record=strax.DEFAULT_RECORD_LENGTH,
                   events_per_chunk=10):
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

    results = []

    def finish_results():
        nonlocal results
        records = np.concatenate(results)
        # In strax data, records are always stored
        # sorted, baselined and integrated
        records = strax.sort_by_time(records)
        strax.baseline(records)
        strax.integrate(records)
        results = []
        return records

    for event in mypax.get_events():
        event = mypax.process_event(event)

        pulse_lengths = np.array([p.length
                                  for p in event.pulses])

        n_records_tot = strax.records_needed(pulse_lengths,
                                             samples_per_record).sum()
        records = np.zeros(n_records_tot,
                           dtype=strax.record_dtype(samples_per_record))
        output_record_index = 0  # Record offset in data

        for p in event.pulses:
            n_records = strax.records_needed(p.length, samples_per_record)

            for rec_i in range(n_records):
                r = records[output_record_index]
                r['time'] = (event.start_time
                             + p.left * 10
                             + rec_i * samples_per_record * 10)
                r['channel'] = p.channel
                r['pulse_length'] = p.length
                r['record_i'] = rec_i
                r['dt'] = 10

                # How much are we storing in this record?
                if rec_i != n_records - 1:
                    # There's more chunks coming, so we store a full chunk
                    n_store = samples_per_record
                    assert p.length > samples_per_record * (rec_i + 1)
                else:
                    # Just enough to store the rest of the data
                    # Note it's not p.length % samples_per_record!!!
                    # (that would be zero if we have to store a full record)
                    n_store = p.length - samples_per_record * rec_i

                assert 0 <= n_store <= samples_per_record
                r['length'] = n_store

                offset = rec_i * samples_per_record
                r['data'][:n_store] = p.raw_data[offset:offset + n_store]
                output_record_index += 1

        results.append(records)
        if len(results) >= events_per_chunk:
            yield finish_results()

    mypax.shutdown()

    if len(results):
        yield finish_results()


@export
@strax.takes_config(
    strax.Option('pax_raw_dir', default='/data/xenon/raw', track=False,
                 help="Directory with raw pax datasets"),
    strax.Option('stop_after_zips', default=0, track=False,
                 help="Convert only this many zip files. 0 = all."),
    strax.Option('events_per_chunk', default=50, track=False,
                 help="Number of events to yield per chunk")
)
class RecordsFromPax(strax.Plugin):
    provides = 'raw_records'
    data_kind = 'raw_records'
    depends_on = tuple()
    dtype = strax.record_dtype()
    parallel = False
    rechunk_on_save = False

    def iter(self, *args, **kwargs):
        if not os.path.exists(self.config['pax_raw_dir']):
            raise FileNotFoundError(self.config['pax_raw_dir'])
        input_dir = os.path.join(self.config['pax_raw_dir'], self.run_id)
        pax_files = sorted(glob.glob(input_dir + '/XENON*.zip'))
        pax_sizes = np.array([os.path.getsize(x)
                              for x in pax_files])
        print(f"Found {len(pax_files)} files, {pax_sizes.sum() / 1e9:.2f} GB")
        for file_i, in_fn in enumerate(pax_files):
            if (self.config['stop_after_zips']
                    and file_i >= self.config['stop_after_zips']):
                break
            yield from strax.xenon.pax_interface.pax_to_records(in_fn)
