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
parser.add_argument('--rate', default=0, type=int,
                    help='Output rate in MBraw/sec. '
                         'If omitted, emit data as if in realtime.')
parser.add_argument('--shm', action='store_true',
                    help='Operate in /dev/shm')
parser.add_argument('--chunk_duration', default=2, type=int,
                    help='Chunk size in sec')
parser.add_argument('--sync_chunk_duration', default=0.2, type=int,
                    help='Synchronization chunk size in sec')
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

st = strax.Context(config=dict(stop_after_zips=5))   # Maybe not track for now...
st.register(strax.xenon.pax_interface.RecordsFromPax)


@numba.njit
def restore_baseline(records):
    for r in records:
        r['data'][:r['length']] = 16000 - r['data'][:r['length']]


def write_to_dir(c, outdir):
    tempdir = outdir + '_temp'
    os.makedirs(tempdir)
    for reader_i, x in enumerate(c):
        with open(f'{tempdir}/reader_{reader_i}', 'wb') as f:
            f.write(copy(x))        # Copy needed for honest shm writing?
    os.rename(tempdir, outdir)


chunk_sizes = []
chunk_data_compressed = []
time_offset = int(time.time()) * int(1e9)
for chunk_i, records in enumerate(
        strax.alternating_duration_chunks(
            st.get_iter('180423_1021', 'raw_records'),
            args.chunk_duration * 1e9,
            args.sync_chunk_duration * 1e9)):
    t_0 = time.time()

    if chunk_i == 0:
        payload_t_start = records[0]['time']
    payload_t_end = records[-1]['time']

    # Restore baseline, clear metadata, fix time
    restore_baseline(records)
    records['baseline'] = 0
    records['area'] = 0
    records['time'] += time_offset - payload_t_start

    chunk_sizes.append(records.nbytes)
    result = []
    for reader_i in range(n_readers):
        first_channel = reader_i * channels_per_reader
        r = records[
                (records['channel'] >= first_channel)
                & (records['channel'] < first_channel + channels_per_reader)]
        r = strax.io.COMPRESSORS['zstd']['compress'](r)
        result.append(r)

    big_chunk_i = chunk_i // 2

    if chunk_i % 2 != 0:
        dt = args.sync_chunk_duration
        write_to_dir(result, output_dir + '/%06d_post' % big_chunk_i)
        write_to_dir(result, output_dir + '/%06d_pre' % (big_chunk_i + 1))
    else:
        dt = args.chunk_duration
        write_to_dir(result, output_dir + '/%06d' % big_chunk_i)

    wrote_mb = chunk_sizes[chunk_i] / 1e6

    if args.rate:
        t_sleep = wrote_mb / args.rate - (time.time() - t_0)
    else:
        t_sleep = dt - (time.time() - t_0)

    if t_sleep < 0:
        print("Fake DAQ too slow :-(")
    else:
        print(f"{chunk_i}: wrote {wrote_mb:.1f} MB_raw, "
              f"sleep for {t_sleep:.2f} s")
        time.sleep(t_sleep)


with open(output_dir + '/THE_END', mode='w') as f:
    f.write("That's all folks!")
