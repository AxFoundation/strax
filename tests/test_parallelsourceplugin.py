import tempfile
import pytest
import strax
import numpy as np

# TODO: these are small modifications of the test helpers in test_core.py
# Can we avoid duplication somehow?
n_chunks = 10
recs_per_chunk = 10
run_id = '0'


class SomeCrash(Exception):
    pass


@strax.takes_config(
    strax.Option('crash', default=False)
)
class Records(strax.ParallelSourcePlugin):
    provides = 'records'
    depends_on = tuple()
    dtype = strax.record_dtype()

    def compute(self, chunk_i):
        if self.config['crash']:
            raise SomeCrash("CRASH!!!!")
        r = np.zeros(recs_per_chunk, self.dtype)
        r['time'] = chunk_i
        r['length'] = 1
        r['dt'] = 1
        r['channel'] = np.arange(len(r))
        return r

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < n_chunks


class Peaks(strax.Plugin):
    parallel = True
    provides = 'peaks'
    depends_on = ('records',)
    dtype = strax.peak_dtype()

    def compute(self, records):
        assert isinstance(records, np.ndarray), \
            f"Recieved {type(records)} instead of numpy array!"
        p = np.zeros(len(records), self.dtype)
        p['time'] = records['time']
        return p


def test_processing():
    """Test ParallelSource plugin under several conditions"""
    # It's always harder with a small mailbox:
    strax.Mailbox.DEFAULT_MAX_MESSAGES = 2
    for request_peaks in (True, False):
        for peaks_parallel in (True, False):
            for max_workers in (1, 2):
                Peaks.parallel = peaks_parallel
                print(f"\nTesting with request_peaks {request_peaks}, "
                      f"peaks_parallel {peaks_parallel}, "
                      f"max_workers {max_workers}")

                mystrax = strax.Context(storage=[],
                                        register=[Records, Peaks])
                bla = mystrax.get_array(
                    run_id=run_id,
                    targets='peaks' if request_peaks else 'records',
                    max_workers=max_workers)
                assert len(bla) == recs_per_chunk * n_chunks
                assert bla.dtype == (
                    strax.peak_dtype() if request_peaks else strax.record_dtype())


# TODO: copy-paste-modified from test_core... not so good
def test_exception():
    Peaks.parallel = True
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks],
                           config=dict(crash=True))

        # Check correct exception is thrown
        with pytest.raises(SomeCrash):
            st.make(run_id=run_id, targets='peaks',
                    max_workers=2)

        # Check exception is recorded in metadata
        # in both its original data type and dependents
        for target in ('peaks', 'records'):
            assert 'SomeCrash' in st.get_meta(run_id, target)['exception']

        # Check corrupted data does not load
        st.context_config['forbid_creation_of'] = ('peaks',)
        with pytest.raises(strax.DataNotAvailable):
            st.get_df(run_id=run_id, targets='peaks')


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='{name} in {threadName} at {asctime}: {message}',
        style='{')
    test_processing()
    test_exception()
