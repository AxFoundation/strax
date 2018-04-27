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

run_id = '180219_2005'
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
        erase=args.erase,
        to_pe=strax.xenon.common.to_pe))
mystrax.register_all(strax.xenon.plugins)

gil_load.start(av_sample_interval=0.1)
start = time.time()

for i, p in enumerate(mystrax.get(run_id,
                                  'peak_classification',
                                  max_workers=args.n)):
    n_s1s = (p['type'] == 1).sum()
    print(f"\t\t{i}: Found {n_s1s} S1s")

end = time.time()
gil_load.stop()

# Get the filesize from the metadata
with open(f'{out_dir}/{run_id}_records/metadata.json', mode='r') as f:
    metadata = json.loads(f.read())
raw_data_size = sum(x['nbytes'] for x in metadata['chunks'])

dt = end - start
speed = raw_data_size / dt / 1e6
print(f"Took {dt:.3} seconds, processing speed was {speed:.4} MB/s")
gil_pct = 100 * gil_load.get(4)[0]
print(f"GIL was held {gil_pct:.3f}% of the time")