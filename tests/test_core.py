"""A very basic test for the strax core.
Mostly tests if we don't crash immediately..
"""
import tempfile
import os
import os.path as osp
import glob

import pytest
import numpy as np
import strax


@strax.takes_config(
    strax.Option('crash', default=False)
)
class Records(strax.Plugin):
    provides = 'records'
    depends_on = tuple()
    dtype = strax.record_dtype()

    def iter(self, *args, **kwargs):
        if self.config['crash']:
            raise SomeCrash("CRASH!!!!")
        for t in range(n_chunks):
            r = np.zeros(recs_per_chunk, self.dtype)
            r['time'] = t
            r['length'] = 1
            r['dt'] = 1
            r['channel'] = np.arange(len(r))
            yield r


class SomeCrash(Exception):
    pass


@strax.takes_config(
    strax.Option('some_option', default=0)
)
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
run_id = '0'


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

        assert not mystrax.is_stored(run_id, 'peaks')
        mystrax.make(run_id=run_id, targets='peaks')
        assert mystrax.is_stored(run_id, 'peaks')

        # We should have two directories
        data_dirs = sorted(glob.glob(osp.join(temp_dir, '*/')))
        assert len(data_dirs) == 2

        # The first dir contains peaks.
        # It should have one data chunk (rechunk is on) and a metadata file
        assert os.listdir(data_dirs[0]) == ['000000', 'metadata.json']

        # Check metadata got written correctly.
        metadata = mystrax.get_meta(run_id, 'peaks')
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


def test_fuzzy_matching():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks])

        st.make(run_id=run_id, targets='peaks')

        # Changing option causes data not to match
        st.set_config(dict(some_option=1))
        assert not st.is_stored(run_id, 'peaks')

        # In fuzzy context, data does match
        st2 = st.new_context(fuzzy_for=('peaks',))
        assert st2.is_stored(run_id, 'peaks')

        st2 = st.new_context(fuzzy_for_options=('some_option',))
        assert st2.is_stored(run_id, 'peaks')

        # Can also set fuzzy options in current context
        st.set_context_config(dict(fuzzy_for=('peaks',)))
        assert st.is_stored(run_id, 'peaks')


def test_storage_converter():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks])
        st.make(run_id=run_id, targets='peaks')

        with tempfile.TemporaryDirectory() as temp_dir_2:
            st = strax.Context(
                storage=[strax.DataDirectory(temp_dir, readonly=True),
                         strax.DataDirectory(temp_dir_2)],
                register=[Records, Peaks],
                storage_converter=True)
            store_1, store_2 = st.storage

            # Data is now in store 1, but not store 2
            key = st._key_for(run_id, 'peaks')
            store_1.find(key)
            with pytest.raises(strax.DataNotAvailable):
                store_2.find(key)

            st.make(run_id, 'peaks')

            # Data is now in both stores
            store_1.find(key)
            store_2.find(key)


def test_exception():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks],
                           config=dict(crash=True))

        # Check correct exception is thrown
        with pytest.raises(SomeCrash):
            st.make(run_id=run_id, targets='peaks')

        # Check exception is recorded in metadata
        # in both its original data type and dependents
        for target in ('peaks', 'records'):
            assert 'SomeCrash' in st.get_meta(run_id, target)['exception']


def test_random_access():
    """Test basic random access
    TODO: test random access when time info is not provided directly
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Hack to enable testing if only required chunks are loaded
        Peaks.rechunk_on_save = False

        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks])

        with pytest.raises(strax.DataNotAvailable):
            # Time range selection requires data already available
            st.get_df(run_id, 'peaks', time_range=(3, 5))

        st.make(run_id=run_id, targets='peaks')

        # Second part of hack: corrupt data by removing one chunk
        os.remove(os.path.join(temp_dir,
                               str(st._key_for(run_id, 'peaks')),
                               '000000'))

        with pytest.raises(FileNotFoundError):
            st.get_array(run_id, 'peaks')

        df = st.get_array(run_id, 'peaks', time_range=(3, 5))
        assert len(df) == 2 * recs_per_chunk
        assert df['time'].min() == 3
        assert df['time'].max() == 4
