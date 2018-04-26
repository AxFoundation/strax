import logging
import argparse
import os
import shutil

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


mystrax = strax.Strax(storage=[strax.FileStorage(data_dirs=[out_dir])],
                      max_workers=args.n)
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
    rechunk = False

    def _path(self, chunk_i):
        return f'{self.in_dir}/{chunk_i:06d}'

    def check_next_ready_or_done(self, chunk_i):
        while not os.path.exists(self._path(chunk_i)):
            if os.path.exists(f'{self.in_dir}/THE_END'):
                return False
            self.log.info("Nothing to submit, sleeping")
            time.sleep(2)
        return True

    def compute(self, chunk_i):
        self.log.info(f"\t{chunk_i}: received from readers")
        records = [strax.load_file(fn,
                                   compressor='zstd',
                                   dtype=strax.record_dtype())
                   for fn in glob.glob(f'{self._path(chunk_i)}/reader_*')]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)
        if self.erase:
            shutil.rmtree(self._path(chunk_i))
        return records


for i, p in enumerate(mystrax.get(run_id, 'peak_classification')):
    n_s1s = (p['type'] == 1).sum()
    print(f"{i}: Found {n_s1s} S1s")
