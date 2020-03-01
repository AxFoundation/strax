import tempfile

from strax.testutils import *


class EvenOddSplit(strax.Plugin):
    parallel = True
    depends_on = 'records'
    provides = ('even_recs', 'odd_recs', 'rec_count')

    data_kind = dict(
        even_recs='even_recs',
        odd_recs='odd_recs',
        rec_count='chunk_bonus_data')

    dtype = dict(
        even_recs=Records.dtype,
        odd_recs=Records.dtype,
        rec_count=[
            ('time', np.int64, 'Time of first record in chunk'),
            ('endtime', np.int64, 'Endtime of last record in chunk'),
            ('n_records', np.int32, 'Number of records'),
        ])

    def compute(self, records):
        mask = records['time'] % 2 == 0
        return dict(even_recs=records[mask],
                    odd_recs=records[~mask],
                    rec_count=dict(n_records=[len(records)],
                                   time=records[0]['time'],
                                   endtime=strax.endtime(records[-1])))


class FunnyPeaks(strax.Plugin):
    parallel = True
    provides = 'peaks'
    depends_on = 'even_recs'
    dtype = strax.peak_dtype()

    def compute(self, even_recs):
        p = np.zeros(len(even_recs), self.dtype)
        p['time'] = even_recs['time']
        return p


def test_multi_output():
    for max_workers in [1, 2]:
        with tempfile.TemporaryDirectory() as temp_dir:
            mystrax = strax.Context(
                storage=strax.DataDirectory(temp_dir),
                register=[Records, EvenOddSplit, FunnyPeaks],
                allow_multiprocess=True)
            assert not mystrax.is_stored(run_id, 'rec_count')

            # Create stuff
            funny_ps = mystrax.get_array(
                run_id=run_id,
                targets='peaks',
                max_workers=max_workers)
            assert mystrax.is_stored(run_id, 'peaks')
            assert mystrax.is_stored(run_id, 'even_recs')

            # Peaks are correct
            assert np.all(funny_ps['time'] % 2 == 0)
            assert len(funny_ps) == n_chunks * recs_per_chunk / 2

            # Unnecessary things also got stored
            assert mystrax.is_stored(run_id, 'rec_count')
            assert mystrax.is_stored(run_id, 'odd_recs')

            # Record count is correct
            rec_count = mystrax.get_array(run_id, 'rec_count')
            print(rec_count)
            assert len(rec_count) == n_chunks
            np.testing.assert_array_equal(rec_count['n_records'], recs_per_chunk)

            # Even and odd records are correct
            r_even = mystrax.get_array(run_id, 'even_recs')
            r_odd = mystrax.get_array(run_id, 'odd_recs')
            assert np.all(r_even['time'] % 2 == 0)
            assert np.all(r_odd['time'] % 2 == 1)
            assert len(r_even) == n_chunks * recs_per_chunk / 2
            assert len(r_even) == len(r_odd)
