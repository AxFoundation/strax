import argparse
from copy import copy
import os
import shutil
import time

from tqdm import tqdm
import strax
mystrax = strax.Strax()

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

output_dir = './from_fake_daq'
if args.shm:
    output_dir = '/dev/shm/from_fake_daq'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print("Preparing payload data")
chunk_sizes = []
chunk_data_compressed = []
for c in tqdm(strax.chunk_arrays.fixed_size_chunks(
        mystrax.get('180219_2005', 'records'),
        args.chunk_size * 1e6)):
    chunk_sizes.append(c.nbytes)
    chunk_data_compressed.append([
        strax.io.COMPRESSORS['zstd']['compress'](x)
        for x in strax.daq_interface.reader_split(c)
    ])
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
