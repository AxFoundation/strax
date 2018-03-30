"""Utilities for dealing with streams of numpy (record) arrays
maybe this should become its own package?
"""
__all__ = 'ChunkPacer fixed_size_chunks equal_chunks sync_iters'.split()

import itertools

import numpy as np
from strax.utils import first_index_not_below


class ChunkPacer:

    def __init__(self, source):
        self.source = source
        self.buffer = []
        self.buffer_items = 0

    def _fetch_another(self):
        x = next(self.source)
        self.buffer.append(x)
        self.buffer_items += len(x)

    def _squash_buffer(self):
        """Squash the buffer into a single array using np.concatenate"""
        if len(self.buffer) > 1:
            self.buffer = [np.concatenate(self.buffer)]

        # Sanity check on buffer_items
        if not len(self.buffer):
            assert self.buffer_items == 0
        else:
            assert self.buffer_items == len(self.buffer[0])

    def _take_from_buffer(self, n):
        self._squash_buffer()
        if self.buffer_items == 0:
            raise StopIteration
        b = self.buffer[0]

        n = min(n, len(b))
        result, b = np.split(b, [n])
        self.buffer = [b]
        self.buffer_items = len(b)
        return result

    def get_n(self, n: int):
        """Return array of the next n elements produced by source,
        or (if this is less) as many as the source can still produce.
        """
        try:
            while self.buffer_items < n:
                self._fetch_another()
        except StopIteration:
            pass

        return self._take_from_buffer(n)

    def get_until(self, threshold, func=None):
        """Return remaining elements of source below or at threshold,
        assuming source gives sorted arrays.

        :param func: computation to do on array elements before comparison
        """
        if func is None:
            def func(x):
                return x

        if not len(self.buffer):
            self._fetch_another()
        assert len(self.buffer) == 1

        try:
            while not func(self.buffer[-1][-1]) > threshold:
                self._fetch_another()
        except StopIteration:
            pass

        n = first_index_not_below(func(self.buffer[-1]), threshold)
        n += sum(len(x) for x in self.buffer[:-1])
        return self._take_from_buffer(n)

    def put_back_at_start(self, x):
        self.buffer = [x] + self.buffer
        self.buffer_items += 1
        self._squash_buffer()


def fixed_length_chunks(source, n=10):
    """Yield arrays of maximum length n"""
    p = ChunkPacer(source)
    try:
        while True:
            yield p.get_n(n)
    except StopIteration:
        return


def fixed_size_chunks(source, n_bytes=int(1e7)):
    """Yield arrays of maximum size n_bytes"""
    p = ChunkPacer(source)

    # Peek at one item to figure out the size
    x = p.get_n(1)
    bytes_per_item = x.nbytes
    p.put_back_at_start(x)

    n = int(n_bytes / bytes_per_item)
    try:
        while True:
            yield p.get_n(n)
    except StopIteration:
        return


def same_length(*sources):
    """Yield tuples of arrays of the same number of items
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    for x in pacemaker:
        yield tuple([x] + [s.get_n(len(x)) for s in others])


def same_stop(*sources, field=None, func=None):
    """Yield tuples of arrays whose values (of field_name) are below common
    thresholds (set by the chunking of sources[0])
    assumes sources are sorted by field (or return value of func)
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    for x in pacemaker:
        threshold = x[-1]
        if field is not None:
            threshold = threshold[field]
        if func is not None:
            threshold = func(threshold)
        yield tuple([x] + [s.get_until(threshold, func=func)
                           for s in others])


def sync_iters(chunker, sources):
    """Return dict of iterators over sources (dict name -> iter),
    synchronized using chunker
    """
    names = list(sources.keys())
    sources = list(sources.values())

    teed = itertools.tee(chunker(*sources),
                         len(sources))

    def get_item(iterable, index):
        for x in iterable:
            yield x[index]

    return {names[i]: get_item(teed[i], i)
            for i in range(len(names))}
