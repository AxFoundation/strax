import logging
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

logging.basicConfig(
   level=logging.INFO,
   format='{name} in {threadName} at {asctime}: {message}', style='{')
log = logging.getLogger()


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    try:
        return subprocess.check_output([
            'du','-sh', path]).split()[0].decode('utf-8')
    except subprocess.CalledProcessError:
        return float('nan')


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
    records = [strax.load_file(fn,
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

log.debug("Created low-level processor")

##
# High-level processing: build everything from peaks
##
@mystrax.register
class TellS1s(strax.Plugin):
    """Stupid plugin to report number of S1s found
    should be removed soon
    """
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

# TODO: make option to NOT save? This is too ugly:
mystrax._plugin_class_registry['peaks'].save_when = strax.SaveWhen.NEVER

high_proc = mystrax.in_background(run_id, 'tell_s1s', sources=['peaks'])


def finish(future, chunk_i):
    high_proc.send('peaks', chunk_i, future.result())
    if not args.erase:
        done_chunks.add(chunk_i)
    pending_chunks.remove(chunk_i)



##
# Main loop
##

done = False
pending_chunks = set()
done_chunks = set()         # Only used if not erasing data (unbounded memory)
high_proc.log.setLevel(logging.DEBUG)
with low_proc, high_proc, ProcessPoolExecutor(max_workers=args.n) as pool:
    while not done:
        print(f'{du(in_dir)} compressed data in buffer, '
              f'{len(pending_chunks)} chunks pending, '
              f'{len(done_chunks)} done.')
        if os.path.exists(in_dir + '/THE_END'):
            done = True

        to_submit = []
        for x in glob.glob(in_dir + '/*/'):
            c = int(x.split('/')[-2])
            if c not in pending_chunks and c not in done_chunks:
                to_submit.append(c)

        if not len(to_submit):
            print("\t\tNothing to submit, sleeping")
            time.sleep(2)
            continue

        for chunk_i in to_submit:
            print(f"\t{chunk_i}: job submitted")
            f = pool.submit(build, chunk_i)
            f.add_done_callback(partial(finish,
                                        chunk_i=int(chunk_i)))
            pending_chunks.add(chunk_i)

    print("Run done and all jobs submitted. Waiting for final results.")
