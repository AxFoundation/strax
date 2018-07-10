"""A very basic test for the strax core.
Mostly tests if we don't crash immediately..
"""
import tempfile
import os
import os.path as osp
import glob

import numpy as np
import strax


class Records(strax.Plugin):
    provides = 'records'
    depends_on = tuple()
    dtype = strax.record_dtype()

    def iter(self, *args, **kwargs):
        for t in range(n_chunks):
            r = np.zeros(recs_per_chunk, self.dtype)
            r['time'] = t
            r['length'] = 1
            r['dt'] = 1
            r['channel'] = np.arange(len(r))
            yield r


class Peaks(strax.Plugin):
    provides = 'peaks'
    depends_on = ('records',)
    dtype = strax.peak_dtype()

    def compute(self, records):
        p = np.zeros(len(records), self.dtype)
        p['time'] = records['time']
        return p


recs_per_chunk = 10
n_chunks = 10
run_id = 'some_run'


def test_core():
    mystrax = strax.Context(storage=[],
                            register=[Records, Peaks])
    bla = mystrax.get_array(run_id=run_id, targets='peaks')
    assert len(bla) == recs_per_chunk * n_chunks
    assert bla.dtype == strax.peak_dtype()


def test_filestore():
    with tempfile.TemporaryDirectory() as temp_dir:
        mystrax = strax.Context(storage=strax.DataDirectory(temp_dir),
                                register=[Records, Peaks])
        mystrax.make(run_id=run_id, targets='peaks')

        # We should have two directories
        data_dirs = sorted(glob.glob(osp.join(temp_dir, '*/')))
        assert len(data_dirs) == 2

        # The first dir contains peaks.
        # It should have one data chunk (rechunk is on) and a metadata file
        assert os.listdir(data_dirs[0]) == ['000000', 'metadata.json']

        # Check metadata got written correctly.
        metadata = mystrax.get_meta('some_run', 'peaks')
        assert len(metadata)
        assert 'writing_ended' in metadata
        assert 'exception' not in metadata
        assert len(metadata['chunks']) == 1

        # Check data gets loaded from cache, not rebuilt
        md_filename = osp.join(data_dirs[0], 'metadata.json')
        mtime_before = osp.getmtime(md_filename)
        df = mystrax.get_array(run_id=run_id, targets='peaks')
        assert len(df) == recs_per_chunk * n_chunks
        assert mtime_before == osp.getmtime(md_filename)

        # Test the zipfile store. Zipping is still awkward...
        zf = osp.join(temp_dir, f'{run_id}.zip')
        strax.ZipDirectory.zip_dir(temp_dir, zf, delete=True)
        assert osp.exists(zf)

        print(temp_dir)
        print(os.listdir(temp_dir))
        mystrax = strax.Context(storage=strax.ZipDirectory(temp_dir),
                                register=[Records, Peaks])
        metadata_2 = mystrax.get_meta(run_id, 'peaks')
        assert metadata == metadata_2
