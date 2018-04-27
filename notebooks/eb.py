import logging
import argparse
import os
import shutil
import gil_load

gil_load.init()

import numpy as np
import strax

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

mystrax = strax.Strax(storage=[strax.FileStorage(data_dirs=[out_dir])])
mystrax.register_all(strax.xenon.plugins)

import glob
import os
import time
import shutil



@mystrax.register
class DAQReader(strax.Plugin):
    provides = 'records'
    dtype = strax.record_dtype()

    erase = args.erase
    in_dir = in_dir

    parallel = 'process'
    rechunk = False

    def _path(self, chunk_i):
        return f'{self.in_dir}/{chunk_i:06d}'

    def check_next_ready_or_done(self, chunk_i):
        while not os.path.exists(self._path(chunk_i)):
            if os.path.exists(f'{self.in_dir}/THE_END'):
                return False
            print("Nothing to submit, sleeping")
            time.sleep(2)
        return True

    def compute(self, chunk_i):
        print(f"{chunk_i}: received from readers")
        records = [strax.load_file(fn,
                                   compressor='zstd',
                                   dtype=strax.record_dtype())
                   for fn in glob.glob(f'{self._path(chunk_i)}/reader_*')]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)
        if self.erase:
            shutil.rmtree(self._path(chunk_i))
        return records


gil_load.start(av_sample_interval=0.05)
start = time.time()

for i, p in enumerate(mystrax.get(run_id,
                                  'peak_classification',
                                  max_workers=args.n)):
    n_s1s = (p['type'] == 1).sum()
    print(f"\t\t{i}: Found {n_s1s} S1s")

end = time.time()
gil_load.stop()

# Get the filesize from the metadata
import json
with open(f'{out_dir}/{run_id}_records/metadata.json', mode='r') as f:
    metadata = json.loads(f.read())
raw_data_size = sum(x['nbytes'] for x in metadata['chunks'])

dt = end - start
speed = raw_data_size / dt / 1e6
print(f"Took {dt:.3} seconds, processing speed was {speed:.4} MB/s")
gil_pct = 100 * gil_load.get(4)[0]
print(f"GIL was held {gil_pct:.3f}% of the time")