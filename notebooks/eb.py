import logging
import argparse
import os
import time
import shutil

import strax

parser = argparse.ArgumentParser(
    description='XENONnT eventbuilder prototype')

parser.add_argument('--input', default='./from_fake_daq',
                    help='Input directory')
parser.add_argument('--output', default='./from_eb',
                    help='Output directory')

parser.add_argument('--n', default=1, type=int,
                    help='Worker processes to start')
parser.add_argument('--shm', action='store_true',
                    help='Operate in /dev/shm')
parser.add_argument('--erase', action='store_true',
                    help='Erase data after reading it. '
                         'Essential for online operation')
parser.add_argument('--debug', action='store_true',
                    help='Activate debug logging')
parser.add_argument('--target', default='event_info',
                    help='Target data type')
parser.add_argument('--norechunk', action='store_true',
                    help='Disable rechunking')

parser.add_argument('--no_super_raw', action='store_true',
                    help='Do not save unreduced raw data')

args = parser.parse_args()

try:
    import gil_load  # noqa
    gil_load.init()
except ImportError:
    from unittest.mock import Mock
    gil_load = Mock()
    gil_load.get = Mock(return_value=[float('nan')])

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='{name} in {threadName} at {asctime}: {message}', style='{')

run_id = '180423_1021'
if args.shm:
    in_dir = '/dev/shm/from_fake_daq'
else:
    in_dir = args.input
out_dir = args.output

# Clean all output dirs. This is of course temporary!
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

if args.no_super_raw:
    strax.xenon.plugins.DAQReader.save_meta_only = True

st = strax.Context(
    storage=strax.DataDirectory(out_dir),
    config=dict(input_dir=in_dir,
                erase=args.erase),
    allow_rechunk=not args.norechunk)

st.register_all(strax.xenon.plugins)

gil_load.start(av_sample_interval=0.05)
start = time.time()

for i, events in enumerate(
        st.get_iter(run_id, args.target,
                    max_workers=args.n)):
    print(f"\t{i}: Found {len(events)} events")

end = time.time()
gil_load.stop()

dt = end - start
gil_pct = 100 * gil_load.get(4)[0]
print(f"Took {dt:.3f} seconds, GIL was held {gil_pct:.3f}% of the time")


def total_size(data_type, raw=False):
    metadata = st.get_meta(run_id, data_type)
    try:
        return sum(x['nbytes' if raw else 'filesize']
                   for x in metadata['chunks']) / 1e6
    except Exception:
        return float('nan')


raw_data_size = round(total_size('raw_records', raw=True))
speed = raw_data_size / dt
to_show = set(['raw_records', args.target])
if args.target != 'raw_records':
    to_show.add('records')
    if args.target != 'records':
        to_show.add('peaks')
sizes = {d: '%0.2f MB' % total_size(d)
         for d in to_show}
print(f"""
Processed {raw_data_size} MB at {speed:.2f} MB/s
Data sizes on disk: {sizes}
""")
