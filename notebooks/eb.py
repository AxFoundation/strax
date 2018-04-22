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

##
# Setup dir
##

if args.shm:
    in_dir = '/dev/shm/from_fake_daq'
    out_dir = '/dev/shm/strax_data'
else:
    in_dir = './from_fake_daq'
    out_dir = './from_eb'

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
else:
    os.makedirs(out_dir)


class Peaks(strax.ReceiverPlugin):
    provides = 'peaks'
    dtype = strax.peak_dtype(n_channels=len(strax.xenon.common.to_pe))


class TellS1s(strax.Plugin):
    # TODO: maybe make empty dtype possible?
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


strax.register_all(strax.xenon.plugins)
mystrax = strax.Strax(storage=[strax.FileStorage(data_dirs=[out_dir])])
mystrax.register(TellS1s)

red_rec_plug = mystrax.provider('reduced_records')
peaks_plug = mystrax.provider('peaks')

mystrax.register(Peaks)
op = mystrax.online(run_id, 'tell_s1s')


def outname(data_type, i):
    dn = f'{out_dir}/{run_id}_{data_type}'
    try:
        os.makedirs(dn)
    except FileExistsError:
        pass
    return dn + ('/%06d' % i)


def build(chunk_i):
    print(f"\t{chunk_i}: started job")
    records = strax.daq_interface.load_from_readers(
        f'{in_dir}/{chunk_i:06d}',
        erase=args.erase)

    # TODO: fix raw save calls. Should be better way to save stuff outside
    # of strax. Also inheritance metadata has to be faked...
    red_rec = red_rec_plug.compute(records=records)
    strax.save(outname('reduced_records', chunk_i),
               red_rec,
               save_meta=False,
               compressor=red_rec_plug.compressor)

    peaks = peaks_plug.compute(reduced_records=red_rec)
    strax.save(outname('peaks', chunk_i),
               peaks,
               save_meta=False,
               compressor=peaks_plug.compressor)
    print(f"\t{chunk_i}: low-level processing done")
    return peaks


def finish(future, chunk_i):
    op.send('peaks', chunk_i, future.result())
    pending_chunks.remove(chunk_i)
    done_chunks.add(chunk_i)



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

op.close()