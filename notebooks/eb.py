import logging
import subprocess
import argparse
import os
import shutil
import time
import glob

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


mystrax = strax.Strax(storage=[strax.FileStorage(data_dirs=[out_dir])],
                      max_workers=args.n)
mystrax.register_all(strax.xenon.plugins)

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
        print(f"\tChunk {self.counter}: done, found {n_s1s} S1s")
        r[0]['n_s1s'] = n_s1s
        return r


processor = mystrax.in_background(run_id, 'tell_s1s', sources=['records'])
log.debug("Created processor")


def from_readers(chunk_i):
    log.info(f"\t{chunk_i}: started job")

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

    pending_chunks.remove(chunk_i)
    log.info(f"\t{chunk_i}: done with job, returning")
    return records


done = False
pending_chunks = set()
with processor:
    while not done:
        print(f'{du(in_dir)} compressed data in buffer.'
              f'{len(pending_chunks)} chunks pending. ')
        if os.path.exists(in_dir + '/THE_END'):
            log.info("Data acquisition stopped")
            done = True

        to_submit = []
        for x in glob.glob(in_dir + '/*/'):
            c = int(x.split('/')[-2])
            if c not in pending_chunks:
                to_submit.append(c)

        if not len(to_submit):
            log.info("\t\tNothing to submit, sleeping")
            time.sleep(2)
            continue

        # TODO: sorted should not be necessary?
        for chunk_i in sorted(to_submit):
            log.info(f"\t{chunk_i}: job submitted")
            f = processor.send(
                'records',
                 data=mystrax.executor.submit(from_readers, chunk_i),
                 chunk_i=chunk_i)
            pending_chunks.add(chunk_i)

    while pending_chunks:
        log.info("\t\tWaiting to read last chunks from readers")
        time.sleep(2)

    log.info("All chunks read in, waiting for final processing")
