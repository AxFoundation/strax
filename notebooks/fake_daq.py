import argparse
import os
from copy import copy
import shutil
import time

import numpy as np
from tqdm import tqdm
import strax

parser = argparse.ArgumentParser(
    description='Fake DAQ to test XENONnT eventbuilder prototype')
parser.add_argument('--rate', default=100, type=int,
                    help='Output rate in MBraw/sec')
parser.add_argument('--shm', action='store_true',
                    help='Operate in /dev/shm')
parser.add_argument('--t', default=10, type=int,
                    help='How long to run the fake DAQ')
parser.add_argument('--chunk_size', default=100, type=int,
                    help='Chunk size in MB')
args = parser.parse_args()

n_readers = 8
n_channels = len(strax.xenon.common.to_pe)
channels_per_reader = np.ceil(n_channels / n_readers)

output_dir = './from_fake_daq'
if args.shm:
    output_dir = '/dev/shm/from_fake_daq'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print("Preparing payload data")
mystrax = strax.Strax()
chunk_sizes = []
chunk_data_compressed = []
for records in tqdm(strax.fixed_size_chunks(
        mystrax.get('180423_1021', 'records'),
        args.chunk_size * 1e6)):

    # Restore baseline, clear metadata
    records['data'] = 16000 - records['data']
    records['baseline'] = 0
    records['area'] = 0

    chunk_sizes.append(records.nbytes)
    result = []
    for reader_i in range(n_readers):
        first_channel = reader_i * channels_per_reader
        r = records[
                (records['channel'] >= first_channel)
                & (records['channel'] < first_channel + channels_per_reader)]
        r = strax.io.COMPRESSORS['zstd']['compress'](r)
        result.append(r)
    chunk_data_compressed.append(result)

print(f"Prepared {len(chunk_sizes)} chunks of "
      f"total size {sum(chunk_sizes)/1e6:.4} MB")

program_start = time.time()
n_chunks_written = 0
done = False
while not done:
    for chunk_i, c in enumerate(chunk_data_compressed):
        t_0 = time.time()

        if t_0 > program_start + args.t:
            with open(output_dir + '/THE_END', mode='w') as f:
                f.write("That's all folks!")
            done = True
            break

        outdir = output_dir + '/%06d' % n_chunks_written
        tempdir = outdir + '_temp'
        os.makedirs(tempdir)
        for reader_i, x in enumerate(c):
            with open(f'{tempdir}/reader_{reader_i}', 'wb') as f:
                f.write(copy(x))
        os.rename(tempdir, outdir)
        wrote_mb = chunk_sizes[chunk_i] / 1e6

        t_sleep = wrote_mb / args.rate - (time.time() - t_0)
        if t_sleep < 0:
            print("Fake DAQ too slow :-(")
        else:
            print(f"Wrote {wrote_mb:.1f} MB_raw, sleep for {t_sleep:.2f} s")
            time.sleep(t_sleep)

        n_chunks_written += 1
