import strax
from strax.testutils import Records, Peaks, run_id
import tempfile
import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import typing as ty
import os

def _get_context(temp_dir=tempfile.gettempdir()):
    st = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                   deep_scan=True),
                       register=[Records, Peaks])
    return st


def test_search_field():
    st = _get_context()
    df = st.search_field('data')


def test_show_config():
    st = _get_context()
    df = st.show_config('peaks')
    assert len(df)


def test_data_info():
    st = _get_context()
    df = st.data_info('peaks')
    assert len(df)


def test_copy_to_frontend():
    """
    Write some data, add a new storage frontend and make sure that our
        copy to that frontend is succesful
    """
    # We need two directories for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as temp_dir_2:
            st = _get_context(temp_dir)
            # Make some data
            st.get_array(run_id, 'records')
            assert st.is_stored(run_id, 'records')

            # Add the second frontend
            st.storage += [strax.DataDirectory(temp_dir_2)]
            st.copy_to_frontend(run_id, 'records',
                                target_compressor='lz4')

            # Make sure both frontends have the same data.
            assert os.listdir(temp_dir) == os.listdir(temp_dir)
            rec_folder = os.listdir(temp_dir)[0]
            assert (
                    os.listdir(os.path.join(temp_dir, rec_folder)) ==
                    os.listdir(os.path.join(temp_dir_2, rec_folder))
            )
