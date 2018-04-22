# import logging
# logging.basicConfig(level=logging.DEBUG,
#                     format='{name} in {threadName} at {asctime}: {message}', style='{')
# logging.getLogger().setLevel(logging.DEBUG)
import subprocess
import argparse
import os
import shutil
import time
import glob
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
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
args = parser.parse_args()

run_id = '180219_2005'
in_dir = '/dev/shm/from_fake_daq' if args.shm else './from_fake_daq'
out_dir = './from_eb'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
else:
    os.makedirs(out_dir)


##
# Low-level processing: process records from readers
# to reduced records and peaks,
# without the overhead of strax's threading system.
##
mystrax = strax.Strax(storage=[strax.FileStorage(data_dirs=[out_dir])])
mystrax.register_all(strax.xenon.plugins)
low_proc = mystrax.simple_chain(run_id, source='records', target='peaks')


def build(chunk_i):
    print(f"\t{chunk_i}: started job")

    # Concatenate data from readers
    chunk_dir_path = f'{in_dir}/{chunk_i:06d}'
    records = [strax.load(fn,
                          compressor='zstd',
                          dtype=strax.record_dtype())
               for fn in glob.glob(f'{chunk_dir_path}/reader_*')]
    records = np.concatenate(records)
    records = strax.sort_by_time(records)
    if args.erase:
        shutil.rmtree(chunk_dir_path)

    peaks = low_proc.send(chunk_i=chunk_i, data=records)
    print(f"\t{chunk_i}: low-level processing done")
    return peaks


##
# High-level processing: build everything from peaks
##
class PeakFeeder(strax.ReceiverPlugin):
    provides = 'peaks'
    dtype = strax.peak_dtype(n_channels=len(strax.xenon.common.to_pe))
    save_when = strax.SaveWhen.NEVER


class TellS1s(strax.Plugin):
    """Stupid plugin to report number of S1s found"""
    dtype = [('n_s1s', np.int16)]
    counter = 0
    parallel = False
    save_when = strax.SaveWhen.NEVER

    def compute(self, peaks, peak_classification):
        p = peak_classification
        r = np.zeros(1, dtype=self.dtype)
        n_s1s = (p['type'] == 1).sum()
        self.counter += 1
        print(f"\t{self.counter}: High-level processing done, "
              f"found {n_s1s} S1s")
        r[0]['n_s1s'] = n_s1s
        return r

mystrax.register(PeakFeeder)
mystrax.register(TellS1s)
high_proc = mystrax.online(run_id, 'tell_s1s')


def finish(future, chunk_i):
    high_proc.send('peaks', chunk_i, future.result())
    pending_chunks.remove(chunk_i)
    done_chunks.add(chunk_i)


##
# Main loop
##


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    try:
        return subprocess.check_output([
            'du','-sh', path]).split()[0].decode('utf-8')
    except subprocess.CalledProcessError:
        return float('nan')


done = False
pending_chunks = set()
done_chunks = set()         # Hmm, clean up once in a while?
with ProcessPoolExecutor(max_workers=args.n) as pool:
    while not done:
        print(f'{du(in_dir)} compressed data in buffer, '
              f'{len(pending_chunks)} chunks pending, '
              f'{len(done_chunks)} done.')
        if os.path.exists(in_dir + '/THE_END'):
            done = True

        chunks = [int(x)
                  for x in sorted(os.listdir(in_dir))
                  if not x.endswith('_temp')
                  and os.path.isdir(in_dir + '/' + x)]
        chunks = [c for c in chunks
                  if c not in pending_chunks
                  and c not in done_chunks]
        if not len(chunks):
            print("\t\tNothing to submit, sleeping")
            time.sleep(2)
            continue

        for chunk_i in chunks:
            print(f"\t{chunk_i}: job submitted")
            # Submit new job
            f = pool.submit(build, chunk_i)
            f.add_done_callback(partial(finish, chunk_i=int(chunk_i)))
            pending_chunks.add(chunk_i)

print("Run done and all jobs submitted. Waiting for final results.")
high_proc.close()
low_proc.close()
