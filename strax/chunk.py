import typing as ty

import numpy as np
import numba

import strax
export, __all__ = strax.exporter()
__all__ += ['default_chunk_size_mb']


default_chunk_size_mb = 200

@export
class Chunk:
    """Single chunk of strax data of one data type"""

    data_type: str
    data_kind: str
    dtype: np.dtype

    # run_id is not superfluous to track:
    # this could change during the run in superruns (in the future)
    run_id: str
    start: int
    end: int

    data: np.ndarray
    target_size_mb: int

    def __init__(self,
                 *,
                 data_type,
                 data_kind,
                 dtype,
                 run_id,
                 start,
                 end,
                 data,
                 target_size_mb=default_chunk_size_mb):
        self.data_type = data_type
        self.data_kind = data_kind
        self.dtype = np.dtype(dtype)
        self.run_id = run_id
        self.start = start
        self.end = end
        if data is None:
            data = np.empty(0, dtype)
        self.data = data
        self.target_size_mb = target_size_mb

        if not (isinstance(self.start, (int, np.integer))
                and isinstance(self.end, (int, np.integer))):
            raise ValueError(f"Attempt to create chunk {self} "
                             "with non-integer start times")
        # Convert to bona fide python integers
        self.start = int(self.start)
        self.end = int(self.end)

        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Attempt to create chunk {self} "
                             "with data that isn't a numpy array")
        expected_dtype = strax.remove_titles_from_dtype(dtype)
        got_dtype = strax.remove_titles_from_dtype(dtype)
        if expected_dtype != got_dtype:
            raise ValueError(f"Attempt to create chunk {self} "
                             f"with data of {dtype}, "
                             f"should be {expected_dtype}")
        if self.start > self.end:
            raise ValueError(f"Attempt to create chunk {self} "
                             f"with negative length")

        if len(self.data):
            data_starts_at = self.data[0]['time']
            # Check the last 500 samples (arbitrary number) as sanity check
            data_ends_at = strax.endtime(self.data[-500:]).max()

            if data_starts_at < self.start:
                raise ValueError(f"Attempt to create chunk {self} "
                                 f"whose data starts early at {data_starts_at}")
            if data_ends_at > self.end:
                raise ValueError(f"Attempt to create chunk {self} "
                                 f"whose data ends late at {data_ends_at}")

        # This is commented out for performance, but it's perhaps useful
        # when debugging
        # if len(data) > 1:
        #     if min(np.diff(data['time'])) < 0:
        #         raise ValueError(f"Attempt to create chunk {self} "
        #                          "whose data is not sorted by time.")

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _t_fmt(t):
        return f'{t // int(1e9)}sec {t % int(1e9)} ns'

    def __repr__(self):
        return (
                f"[{self.run_id}.{self.data_type}: "
                f"{self._t_fmt(self.start)} - {self._t_fmt(self.end)}, "
                f"{len(self)} items, " +
                "{0:.1f} MB/s]".format(self._mbs()))

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def duration(self):
        return self.end - self.start

    def _mbs(self):
        if self.duration:
            return (self.nbytes / 1e6) / (self.duration / 1e9)
        else:
            # This is strange. We have a zero duration chunk. However, this is
            # not the right place to raise an error message. Return -1 for now.
            return -1

    def split(self,
              t: ty.Union[int, None],
              allow_early_split=False):
        """Return (chunk_left, chunk_right) split at time t.

        :param t: Time at which to split the data.
        All data in the left chunk will have their (exclusive) end <= t,
        all data in the right chunk will have (inclusive) start >=t.
        :param allow_early_split:
          If False, raise CannotSplit if the requirements above cannot be met.
          If True, split at the closest possible time before t.
        """
        t = max(min(t, self.end), self.start)
        if t == self.end:
            data1, data2 = self.data, self.data[:0]
        elif t == self.start:
            data1, data2 = self.data[:0], self.data
        else:
            data1, data2, t = split_array(
                data=self.data,
                t=t,
                allow_early_split=allow_early_split)

        common_kwargs = dict(
            run_id=self.run_id,
            dtype=self.dtype,
            data_type=self.data_type,
            data_kind=self.data_kind,
            target_size_mb=self.target_size_mb)

        c1 = strax.Chunk(
            start=self.start,
            end=max(self.start, t),
            data=data1,
            **common_kwargs)
        c2 = strax.Chunk(
            start=max(self.start, t),
            end=max(t, self.end),
            data=data2,
            **common_kwargs)
        return c1, c2

    @classmethod
    def merge(cls, chunks, data_type='<UNKNOWN>'):
        """Create chunk by merging columns of chunks of same data kind

        :param chunks: Chunks to merge. None is allowed and will be ignored.
        :param data_type: data_type name of new created chunk. Set to <UNKNOWN>
        if not provided.
        """
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise ValueError("Need at least one chunk to merge")
        if len(chunks) == 1:
            return chunks[0]

        data_kinds = [c.data_kind for c in chunks]
        if len(set(data_kinds)) != 1:
            raise ValueError(f"Cannot merge chunks {chunks} of different"
                             f" data kinds: {data_kinds}")
        data_kind = data_kinds[0]

        run_ids = [c.run_id for c in chunks]
        if len(set(run_ids)) != 1:
            raise ValueError(
                f"Cannot merge chunks of different run_ids: {chunks}")
        run_id = run_ids[0]

        if len(set([len(c) for c in chunks])) != 1:
            raise ValueError(
                f"Cannot merge chunks with different number of items: {chunks}")

        tranges = [(c.start, c.end) for c in chunks]
        if len(set(tranges)) != 1:
            raise ValueError("Cannot merge chunks with different time "
                             f"ranges: {tranges}")
        start, end = tranges[0]

        data = strax.merge_arrs(
            [c.data for c in chunks],
            # Make sure dtype field order is consistent, regardless of the
            # order in which chunks are passed to merge:
            dtype=strax.merged_dtype(
                [c.dtype
                 for c in sorted(chunks,
                                 key=lambda x: x.data_type)]))

        return cls(
            start=start,
            end=end,
            dtype=data.dtype,
            data_type=data_type,
            data_kind=data_kind,
            run_id=run_id,
            data=data,
            target_size_mb=max([c.target_size_mb for c in chunks]))

    @classmethod
    def concatenate(cls, chunks):
        """Create chunk by concatenating chunks of same data type
        You can pass None's, they will be ignored
        """
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise ValueError("Need at least one chunk to concatenate")
        if len(chunks) == 1:
            return chunks[0]

        data_types = [c.data_type for c in chunks]
        if len(set(data_types)) != 1:
            raise ValueError(f"Cannot concatenate chunks of different "
                             f"data types: {data_types}")
        data_type = data_types[0]

        run_ids = [c.run_id for c in chunks]
        if len(set(run_ids)) != 1:
            raise ValueError(
                f"Cannot concatenate {data_type} chunks with "
                f"different run ids: {run_ids}")
        run_id = run_ids[0]

        prev_end = 0
        for c in chunks:
            if c.start < prev_end:
                raise ValueError(
                    "Attempt to concatenate overlapping or "
                    f"out-of-order chunks: {chunks} ")
            prev_end = c.end

        return cls(
            start=chunks[0].start,
            end=chunks[-1].end,
            dtype=chunks[0].dtype,
            data_type=data_type,
            data_kind=chunks[0].data_kind,
            run_id=run_id,
            data=np.concatenate([c.data for c in chunks]),
            target_size_mb=max([c.target_size_mb for c in chunks]))


@export
def continuity_check(chunk_iter):
    """Check continuity of chunks yielded by chunk_iter as they are yielded"""
    last_end = None
    last_runid = None
    for s in chunk_iter:
        if s.run_id != last_runid:
            # TODO: can we do better?
            last_end = None
        if last_end is not None:
            if s.start != last_end:
                raise ValueError("Data is not continuous. "
                                 f"Chunk {s} should have started at {last_end}")
        yield s

        last_end = s.end
        last_runid = s.run_id


@export
class CannotSplit(Exception):
    pass


@export
@numba.njit(cache=True, nogil=True)
def split_array(data, t, allow_early_split=False):
    """Return (data left of t, data right of t, t), or raise CannotSplit
    if that would split a data element in two.

    :param data: strax numpy data
    :param t: Time to split data
    :param allow_early_split: Instead of raising CannotSplit,
    split at t_split as close as possible before t where a split can happen.
    The new split time replaces t in the return value.
    """
    # Slitting an empty array is easy
    if not len(data):
        return data[:0], data[:0], t

    # Splitting off a bit of nothing from the start is easy
    # since the data is sorted by time.
    if data[0]['time'] >= t:
        return data[:0], data, t

    # Find:
    #  i_first_beyond: the first element starting after t
    #  splittable_i: nearest index left of t where we can safely split BEFORE
    latest_end_seen = -1
    splittable_i = 0
    i_first_beyond = -1
    for i, d in enumerate(data):
        if d['time'] >= latest_end_seen:
            splittable_i = i
        if d['time'] >= t:
            i_first_beyond = i
            break
        latest_end_seen = max(latest_end_seen,
                              strax.endtime(d))
        if latest_end_seen > t:
            # Cannot split anywhere after this
            break
    else:
        if latest_end_seen <= t:
            return data, data[:0], t

    if (splittable_i != i_first_beyond or
            latest_end_seen > t):
        if not allow_early_split:
            # Raise custom exception, make better one outside numba
            raise CannotSplit()
        t = min(data[splittable_i]['time'], t)

    return data[:splittable_i], data[splittable_i:], t
