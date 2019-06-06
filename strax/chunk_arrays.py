"""Utilities for dealing with streams of numpy (record) arrays
"""
import itertools

import numpy as np
import strax
export, __all__ = strax.exporter()


@export
class ChunkPacer:

    def __init__(self, source, dtype=None):
        self.dtype = dtype
        self.source = source
        self.buffer = []
        self.buffer_items = 0
        self._source_exhausted = False

        if self.dtype is None:
            # Peek at one item to figure out the dtype
            self.dtype = self.peek().dtype

    @property
    def exhausted(self):
        return self._source_exhausted and self.buffer_items == 0

    def _fetch_another(self):
        try:
            x = next(self.source)
        except StopIteration:
            self._source_exhausted = True
            raise StopIteration
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
            return np.empty(0, dtype=self.dtype)
        b = self.buffer[0]

        n = min(n, len(b))
        result, b = np.split(b, [n])
        self.buffer = [b]
        self.buffer_items = len(b)
        return result

    def get_n(self, n: int):
        """Return array of the next n elements produced by source,
        or (if this is less) as many as the source can still produce.
        
        raises 
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
            while (not len(self.buffer[-1])
                    or not func(self.buffer[-1][-1]) > threshold):
                self._fetch_another()
        except StopIteration:
            pass

        n = strax.first_index_not_below(func(self.buffer[-1]), threshold)
        n += sum(len(x) for x in self.buffer[:-1])
        return self._take_from_buffer(n)

    def _put_back_at_start(self, x):
        self.buffer = [x] + self.buffer
        self.buffer_items += 1
        self._squash_buffer()

    def peek(self, n=1):
        x = self.get_n(n)
        self._put_back_at_start(x)
        return x

    @property
    def itemsize(self):
        return np.zeros(1, dtype=self.dtype).nbytes


@export
def fixed_length_chunks(source, n=10):
    """Yield arrays of maximum length n"""
    p = ChunkPacer(source)
    while not p.exhausted:
        yield p.get_n(n)


@export
def fixed_size_chunks(source, n_bytes=int(1e8), dtype=None):
    """Yield arrays of maximum size n_bytes"""
    p = ChunkPacer(source, dtype=dtype)
    n = int(n_bytes / p.itemsize)
    while not p.exhausted:
        yield p.get_n(n)


@export
def alternating_size_chunks(source, *sizes):
    """Yield arrays of sizes[0], then sizes[1], ... sizes[n],
    then sizes[0], etc."""
    p = ChunkPacer(source)
    ns = np.floor(np.array(sizes) / p.itemsize).astype(np.int)
    i = 0
    while not p.exhausted:
        yield p.get_n(ns[i])
        i = (i + 1) % len(sizes)


@export
def alternating_duration_chunks(source, *durations):
    """Yield arrays of sizes[0], then sizes[1], ... sizes[n],
    then sizes[0], etc."""
    p = ChunkPacer(source)
    t = p.peek()[0]['time']
    i = 0
    while not p.exhausted:
        t += durations[i]
        yield p.get_until(t, func=strax.endtime)
        i = (i + 1) % len(durations)


@export
def same_length(*sources):
    """Yield tuples of arrays of the same number of items
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    for x in pacemaker:
        yield tuple([x] + [s.get_n(len(x)) for s in others])


@export
def same_stop(*sources, field=None, func=None):
    """Yield tuples of arrays whose values (of field_name) are below common
    thresholds (set by the chunking of sources[0])
    assumes sources are sorted by field (or return value of func)
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    for x in pacemaker:
        if not len(x):
            yield tuple([x] + [np.empty(0, dtype=s.dtype)
                               for s in others])
            continue

        threshold = x[-1]
        if field is not None:
            threshold = threshold[field]
        if func is not None:
            threshold = func(threshold)
        yield tuple([x] + [s.get_until(threshold, func=func)
                           for s in others])


@export
def sync_iters(chunker, sources):
    """Return dict of iterators over sources (dict name -> iter),
    synchronized using chunker.

    If only one array iter is provided, assume no syncing is needed
    """
    names = list(sources.keys())
    sources = list(sources.values())
    if len(sources) == 1:
        return {names[0]: sources[0]}

    teed = itertools.tee(chunker(*sources),
                         len(sources))

    def get_item(iterable, index):
        for x in iterable:
            yield x[index]

    return {names[i]: get_item(teed[i], i)
            for i in range(len(names))}


@export
def merge_iters(iters):
    """Return iterator over merged arrays from several iterators
    :param iters: list, tuple, or dict of iters

    Iterators must already be synced to produce same-size chunks
    """
    if isinstance(iters, dict):
        iters = list(iters.values())
    iters = list(iters)

    if len(iters) == 1:
        yield from iters[0]

    try:
        while True:
            yield strax.merge_arrs([next(it)
                                    for it in iters])
    except StopIteration:
        return
