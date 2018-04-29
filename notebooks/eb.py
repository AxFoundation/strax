import logging
import argparse
import os
import time
import json
import shutil

import gil_load
import strax

gil_load.init()


logging.basicConfig(
    level=logging.INFO,
    format='{name} in {threadName} at {asctime}: {message}', style='{')
log = logging.getLogger()

parser = argparse.ArgumentParser(
    description='XENONnT eventbuilder prototype')
parser.add_argument('--n', default=1, type=int,
                    help='Worker processes to start')
parser.add_argument('--shm', action='store_true',
                    help='Operate in /dev/shm')
parser.add_argument('--erase', action='store_true',
                    help='Erase data after reading it. '
                         'Essential for online operation')
args = parser.parse_args()

run_id = '180423_1021'
in_dir = '/dev/shm/from_fake_daq' if args.shm else './from_fake_daq'
out_dir = './from_eb'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
else:
    os.makedirs(out_dir)

mystrax = strax.Strax(
    storage=[strax.FileStorage(data_dirs=[out_dir])],
    config=dict(
        input_dir=in_dir,
        erase=args.erase))
mystrax.register_all(strax.xenon.plugins)

gil_load.start(av_sample_interval=0.05)
start = time.time()

for i, events in enumerate(
        mystrax.get(run_id, 'event_basics', max_workers=args.n)):
    print(f"\t{i}: Found {len(events)} events")

end = time.time()
gil_load.stop()

def total_size(data_type, raw=False):
    with open(f'{out_dir}/{run_id}_{data_type}/metadata.json', mode='r') as f:
        metadata = json.loads(f.read())
    return sum(x['nbytes' if raw else 'filesize']
               for x in metadata['chunks']) / 1e6

raw_data_size = total_size('records', raw=True)
dt = end - start
speed = raw_data_size / dt
print(f"Took {dt:.3f} seconds, "
      f"processed {round(raw_data_size)} MB at {speed:.2f} MB/s")
gil_pct = 100 * gil_load.get(4)[0]
print(f"GIL was held {gil_pct:.3f}% of the time")
print(f"Data size on disk: ",
      {d: '%0.2f MB' % total_size(d)
       for d in ['records', 'reduced_records',
                 'peaks', 'peak_classification']})
