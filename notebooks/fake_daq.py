import argparse
import os
from copy import copy
import shutil
import time

import numba
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
parser.add_argument('--sync_chunk_size', default=5, type=int,
                    help='Synchronization chunk size in MB')
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

st = strax.Strax(config=dict(stop_after_zips=5))   # Maybe not track for now...
st.register(strax.xenon.pax_interface.RecordsFromPax)


@numba.njit
def restore_baseline(records):
    for r in records:
        r['data'][:r['length']] = 16000 - r['data'][:r['length']]


print("Preparing payload data")
chunk_sizes = []
chunk_data_compressed = []
for records in tqdm(strax.alternating_size_chunks(
        st.get_iter('180423_1021', 'raw_records'),
        args.chunk_size * 1e6,
        args.sync_chunk_size * 1e6)):

    # Restore baseline, clear metadata
    restore_baseline(records)
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

def write_to_dir(c, outdir):
    tempdir = outdir + '_temp'
    os.makedirs(tempdir)
    for reader_i, x in enumerate(c):
        with open(f'{tempdir}/reader_{reader_i}', 'wb') as f:
            f.write(copy(x))        # Copy needed for honest shm writing?
    os.rename(tempdir, outdir)


program_start = time.time()
for chunk_i, c in enumerate(chunk_data_compressed):
    t_0 = time.time()

    if t_0 > program_start + args.t:
        break

    big_chunk_i = chunk_i // 2

    if chunk_i % 2 != 0:
        write_to_dir(c, output_dir + '/%06d_post' % big_chunk_i)
        write_to_dir(c, output_dir + '/%06d_pre' % (big_chunk_i + 1))
    else:
        write_to_dir(c, output_dir + '/%06d' % big_chunk_i)

    wrote_mb = chunk_sizes[chunk_i] / 1e6

    t_sleep = wrote_mb / args.rate - (time.time() - t_0)
    if t_sleep < 0:
        print("Fake DAQ too slow :-(")
    else:
        print(f"{chunk_i}: wrote {wrote_mb:.1f} MB_raw, "
              f"sleep for {t_sleep:.2f} s")
        time.sleep(t_sleep)


with open(output_dir + '/THE_END', mode='w') as f:
    f.write("That's all folks!")
