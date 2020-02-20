import json

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
    stop: int

    data: np.ndarray
    metadata: dict

    def __init__(self,
                 *,
                 data_type,
                 data_kind,
                 dtype,
                 run_id,
                 start,
                 end,
                 data,
                 metadata=None):
        if metadata is None:
            metadata = dict()
        if data is None:
            data = np.empty(0, dtype)
        self.data_type = data_type
        self.data_kind = data_kind
        self.dtype = dtype
        self.run_id = run_id
        self.start = start
        self.end = end
        self.data = data
        self.metadata = metadata

    @property
    def n_rows(self):
        return len(self.data)

    @property
    def duration(self):
        return self.end - self.start

    def json_metadata(self, **kwargs):
        return json.dumps(
            dict(
                start=self.start,
                end=self.end,
                n_rows=self.n_rows,
                run_id=self.run_id,
                metadata=self.metadata),
            **kwargs)
