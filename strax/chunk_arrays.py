"""Utilities for dealing with streams of numpy (record) arrays
"""
import itertools

import numpy as np
import strax
export, __all__ = strax.exporter()


@export
class ChunkPacer:

    def __init__(self, source):
        self.source = source
        self.buffer = []
        self.buffer_items = 0
        self._source_exhausted = False

        # Will initialize dtype and other things needed to make empty chunk
        x = self.peek()
        self.dtype = x.dtype

    @property
    def exhausted(self):
        return self._source_exhausted and self.buffer_items == 0

    def check_is_exhausted(self):
        """Check thoroughly if we are exhausted, by trying to
        fetch a new chunk.
        """
        try:
            self._fetch_another()
        except StopIteration:
            pass
        return self.exhausted

    def slurp(self):
        while not self._source_exhausted:
            try:
                self._fetch_another()
            except StopIteration:
                return

    def _fetch_another(self):
        # Remember to catch StopIteration if you're using this guy!
        # TODO: custom exception?
        try:
            x = next(self.source)
        except StopIteration:
            self._source_exhausted = True
            raise StopIteration
        assert isinstance(x, strax.Chunk), \
            f"Got {type(x)} instead of a strax Chunk!"

        self.last_time_in_buffer = x.end

        self.buffer.append(x)
        self.buffer_items += len(x)

    def _squash_buffer(self):
        """Squash the buffer into a single chunk using np.concatenate"""
        if len(self.buffer) > 1:
            self.buffer = [strax.Chunk.concatenate(self.buffer)]

        # Sanity check on buffer_items
        if not len(self.buffer):
            assert self.buffer_items == 0
        else:
            assert self.buffer_items == len(self.buffer[0])

    def _take_from_buffer(self, n_items=None, until=None):
        self._squash_buffer()

        if not len(self.buffer):
            raise NotImplementedError("Oops")

        b = self.buffer[0]
        result, b = b.split(n_items=n_items, at=until,
                            extend=True)

        self.buffer = [b]
        self.buffer_items = len(b)
        self.last_time_in_buffer = b.end  # Can change if at > buffer_time
        return result

    def get_n(self, n: int):
        """Return array of the next n elements produced by source,
        or (if this is less) as many as the source can still produce.
        """
        assert isinstance(n, int)
        if n == 0:
            raise NotImplementedError("Cannot get zero elements (yet?)")

        try:
            while self.buffer_items < n:
                self._fetch_another()
        except StopIteration:
            n = self.buffer_items

        return self._take_from_buffer(n)

    def get_until(self, threshold: int):
        """Return elements of source ending before or at threshold,
        assuming source gives sorted arrays.
        """
        assert isinstance(threshold, int)
        try:
            while self.last_time_in_buffer < threshold:
                self._fetch_another()
        except StopIteration:
            pass

        return self._take_from_buffer(until=threshold)

    def _put_back_at_start(self, x):
        self.buffer = [x] + self.buffer
        self.buffer_items += 1
        self._squash_buffer()

    def peek(self, n=1):
        x = self.get_n(n)
        if len(x):
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
def fixed_size_chunks(source, n_bytes=int(1e8)):
    """Yield arrays of maximum size n_bytes"""
    p = ChunkPacer(source)
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
    t = p.peek().start
    i = 0
    while not p.exhausted:
        t += durations[i]
        yield p.get_until(t)
        i = (i + 1) % len(durations)


@export
def same_length(*sources, same_length=True):
    """Yield tuples of arrays of the same number of items

    :param same_length: Crash if the sources do not produce
    items of the same length
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    for x in pacemaker:
        yield tuple([x] + [s.get_n(len(x)) for s in others])

    if same_length:
        for s in others:
            assert s.check_is_exhausted(), f"{s.buffer_items} items left!"


@export
def same_end(*sources):
    """Yield tuples of arrays whose values (of field_name) are below common
    endtime thresholds (set by the chunking of sources[0])
    assumes sources are sorted by endtime
    """
    pacemaker = sources[0]
    others = [ChunkPacer(s) for s in sources[1:]]

    def get_result(pacemaker_chunk, is_last):
        if is_last:
            for x in others:
                x.slurp()
            endtime = max([x.last_time_in_buffer for x in others])
            endtime = max(endtime, pacemaker_chunk.end)

            # Final chunk: get ALL remaining data from others
            other_data = [s.get_until(endtime)
                          for s in others]
            pacemaker_chunk.stop = endtime
        else:
            threshold = pacemaker_chunk.end
            other_data = [s.get_until(threshold)
                          for s in others]

        return tuple([pacemaker_chunk] + other_data)

    # The last chunk takes special handling, but does not announce
    # itself. Hence we must always buffer one pacemaker chunk.
    buffer = None
    for next_chunk in pacemaker:
        if buffer is not None:
            yield get_result(buffer, is_last=False)
        buffer = next_chunk
    yield get_result(buffer, is_last=True)


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

    Iterators must already be synced to produce same-time-range chunks
    """
    if isinstance(iters, dict):
        iters = list(iters.values())
    iters = list(iters)

    if len(iters) == 1:
        yield from iters[0]

    try:
        while True:
            yield strax.Chunk.merge([next(it)
                                     for it in iters])
    except StopIteration:
        return
