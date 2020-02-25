import json
import typing as ty

import numpy as np

import strax
export, __all__ = strax.exporter()


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

    def __init__(self,
                 *,
                 data_type,
                 data_kind,
                 dtype,
                 run_id,
                 start,
                 end,
                 data):
        if data is None:
            data = np.empty(0, dtype)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(data, np.ndarray)
        assert strax.remove_titles_from_dtype(data.dtype) == strax.remove_titles_from_dtype(dtype), f"Cannot create chunk: promised {dtype}, fot {data.dtype}"
        dtype = np.dtype(dtype)
        assert end >= start

        self.data_type = data_type
        self.data_kind = data_kind
        self.dtype = dtype
        self.run_id = run_id
        self.start = start
        self.end = end
        self.data = data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{self.run_id}.{self.data_type}:" \
               f"({self.start}-{self.end}, {len(self)})"

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def duration(self):
        return self.end - self.start

    def split(self,
              n_items: ty.Union[int, None] = None,
              at: ty.Union[int, None] = None,
              extend=False):
        """Split a chunk in two.

        :param n_items: Number of items to put in the first chunk
        :param at: Time at or before which all items in the first chunk
        must end
        :param extend: If True, and at > self.end, will extend the endtime of
        the right chunk to at. Usually it will be self.end.
        :return: 2-tuple of strax.Chunks
        """
        if n_items is None and at is None:
            raise ValueError("Provide either n_items or at")
        if n_items is not None and at is not None:
            raise ValueError("Don't provide both n_items and at")

        if n_items is None:
            n_items = strax.first_true(
                strax.endtime(self.data) > at)

        data1, data2 = np.split(self.data, [n_items])

        if at is None:
            # TODO: say somewhere this only works for disjoint data
            if not len(data1):
                at = self.start
            elif not len(data2):
                at = self.end
            else:
                at = int((strax.endtime(data1[-1]) + data2[0]['time']) // 2)

        common_kwargs = dict(
            run_id=self.run_id,
            dtype=self.dtype,
            data_type=self.data_type,
            data_kind=self.data_kind)

        if at > self.end and not extend:
            at = self.end

        c1 = strax.Chunk(
            start=self.start,
            end=max(self.start, at),
            data=data1,
            **common_kwargs)
        c2 = strax.Chunk(
            # if at < start, second fragment contains everything
            start=max(self.start, at),
            end=max(at, self.end),
            data=data2,
            **common_kwargs)
        return c1, c2

    @classmethod
    def merge(cls, chunks, data_type='<UNKNOWN>'):
        """Create chunk by merging columns of chunks of same data kind

        :param chunks: Chunks to merge
        :param dtype: Numpy dtype of merged chunk. Must be explicitly provided,
        otherwise field order is ambiguous
        :param data_type: data_type name of new created chunk. Set to <UNKNOWN>
        if not provided.
        """
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

        tranges = [(c.start, c.end) for c in chunks]
        if len(set(tranges)) != 1:
            raise ValueError(
                f"Cannot merge chunks with different time ranges: {chunks}")
        start, end = tranges[0]

        # Merge chunks in order of data_type name
        # so the field order is consistent
        # regardless of the order in which chunks they are passed
        chunks = list(sorted(chunks, key=lambda x: x.data_type))

        data = strax.merge_arrs([c.data for c in chunks])

        return cls(
            start=start,
            end=end,
            dtype=data.dtype,
            data_type=data_type,
            data_kind=data_kind,
            run_id=run_id,
            data=data)

    @classmethod
    def concatenate(cls, chunks):
        """Create chunk by concatenating chunks of same data type"""
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
            data=np.concatenate([c.data for c in chunks]))
