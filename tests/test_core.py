"""A very basic test for the strax core.
Mostly tests if we don't crash immediately..
"""

import numpy as np
import strax


def test_core():
    recs_per_chunk = 10
    n_chunks = 10

    class Records(strax.Plugin):
        provides = 'records'
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
        dtype = strax.peak_dtype()

        def compute(self, records):
            p = np.zeros(len(records), self.dtype)
            p['time'] = records['time']
            return p

    mystrax = strax.Strax(storage=[])
    mystrax.register(Records)
    mystrax.register(Peaks)

    bla = mystrax.get_array('some_run', 'peaks')
    assert len(bla) == recs_per_chunk * n_chunks
    assert bla.dtype == strax.peak_dtype()


def test_online():
    recs_per_chunk = 10
    n_chunks = 10

    class Records(strax.ReceiverPlugin):
        provides = 'records'
        dtype = strax.record_dtype()

    n_made = 0

    class Peaks(strax.Plugin):
        provides = 'peaks'
        dtype = strax.peak_dtype()

        def compute(self, records):
            nonlocal n_made
            p = np.zeros(len(records), self.dtype)
            p['time'] = records['time']
            n_made += len(p)
            return p

    mystrax = strax.Strax(storage=[])
    mystrax.register(Records)
    mystrax.register(Peaks)

    op = mystrax.online('some_run', 'peaks')
    for i in range(n_chunks):
        r = np.zeros(recs_per_chunk, strax.record_dtype())
        r['time'] = i
        r['length'] = 1
        r['dt'] = 1
        r['channel'] = np.arange(len(r))
        op.send('records', i, r)

    op.close()
    assert n_made == n_chunks * recs_per_chunk
