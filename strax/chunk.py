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
        assert data.dtype == dtype

        self.data_type = data_type
        self.data_kind = data_kind
        self.dtype = dtype
        self.run_id = run_id
        self.start = start
        self.end = end
        self.data = data

    def __len__(self):
        return len(self.data)

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def duration(self):
        return self.end - self.start

    def json_metadata(self, **kwargs):
        return json.dumps(
            dict(
                start=self.start,
                end=self.end,
                n_rows=len(self),
                run_id=self.run_id),
            **kwargs)

    def split(self, n_items: ty.Union[int, None], at: ty.Union[int, None]):
        if n_items is None and at is None:
            raise ValueError("Provide either n_items or at")
        if n_items is not None and at is not None:
            raise ValueError("Don't provide both n_items and at")

        if n_items is None:
            n_items = strax.first_index_not_below(
                strax.endtime(self.data),
                at)

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

        c1 = strax.Chunk(start=self.start,
                         end=at,
                         data=data1,
                         **common_kwargs)
        c2 = strax.Chunk(start=at,
                         end=self.end,
                         data=data2,
                         **common_kwargs)
        return c1, c2

    @classmethod
    def concatenate(cls, chunks):
        if not chunks:
            raise ValueError("Need at least one chunk to concatenate")
        if len(chunks) == 1:
            return chunks[0]

        data_types = [c.data_type for c in chunks]
        if len(set(data_types)) != 1:
            raise ValueError(
                f"Cannot concatenate chunks of different data types: {data_types}")
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
                    f"out-of-order chunks of {data_type} for run {run_id}. "
                    f"Starts: {[c.start for c in chunks]}, "
                    f"Ends: {[c.end for c in chunks]}.")
            prev_end = c.end

        return cls(
            start=chunks[0].start,
            end=chunks[-1].end,
            dtype=chunks[0].dtype,
            data_type=data_type,
            data_kind=chunks[0].data_kind,
            run_id=run_id,
            data=np.concatenate([c.data for c in chunks]))
