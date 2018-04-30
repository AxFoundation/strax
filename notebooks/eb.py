import logging
import argparse
import os
import time
import shutil

import strax


parser = argparse.ArgumentParser(
    description='XENONnT eventbuilder prototype')
parser.add_argument('--n', default=1, type=int,
                    help='Worker processes to start')
parser.add_argument('--shm', action='store_true',
                    help='Operate in /dev/shm')
parser.add_argument('--erase', action='store_true',
                    help='Erase data after reading it. '
                         'Essential for online operation')
parser.add_argument('--debug', action='store_true',
                    help='Activate debug logging')

args = parser.parse_args()

try:
    import gil_load     # noqa
    gil_load.init()
except ImportError:
    from unittest.mock import Mock
    gil_load = Mock()
    gil_load.get = Mock(return_value=[float('nan')])

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='{name} in {threadName} at {asctime}: {message}', style='{')

run_id = '180423_1021'
in_dir = '/dev/shm/from_fake_daq' if args.shm else './from_fake_daq'
out_dir = './from_eb'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
else:
    os.makedirs(out_dir)

st = strax.Strax(storage=out_dir,
                 config=dict(input_dir=in_dir,
                             erase=args.erase))
st.register_all(strax.xenon.plugins)

gil_load.start(av_sample_interval=0.05)
start = time.time()

for i, events in enumerate(
        st.get_iter(run_id, 'event_basics', max_workers=args.n)):
    print(f"\t{i}: Found {len(events)} events")

end = time.time()
gil_load.stop()


def total_size(data_type, raw=False):
    metadata = st.get_meta(run_id, data_type)
    return sum(x['nbytes' if raw else 'filesize']
               for x in metadata['chunks']) / 1e6


raw_data_size = round(total_size('records', raw=True))
dt = end - start
speed = raw_data_size / dt
gil_pct = 100 * gil_load.get(4)[0]
sizes = {d: '%0.2f MB' % total_size(d)
         for d in ['records', 'reduced_records',
                   'peaks', 'peak_classification']}
print(f"""
Took {dt:.3f} seconds, processed {raw_data_size} MB at {speed:.2f} MB/s
GIL was held {gil_pct:.3f}% of the time
Data sizes on disk: {sizes}
""")