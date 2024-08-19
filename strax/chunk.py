import typing as ty

import numpy as np
import numba

import strax

export, __all__ = strax.exporter()
__all__.extend(["DEFAULT_CHUNK_SIZE_MB", "DEFAULT_CHUNK_SPLIT_NS"])


DEFAULT_CHUNK_SIZE_MB = 200
DEFAULT_CHUNK_SPLIT_NS = 1000


@export
class Chunk:
    """Single chunk of strax data of one data type."""

    data_type: str
    data_kind: str
    dtype: np.dtype

    # run_id is not superfluous to track:
    # this could change during the run in superruns (in the future)
    run_id: str
    subruns: dict
    superrun: dict
    start: int
    end: int

    data: np.ndarray
    target_size_mb: int

    def __init__(
        self,
        *,
        data_type,
        data_kind,
        dtype,
        run_id,
        start,
        end,
        data,
        subruns=None,
        superrun=None,
        target_size_mb=DEFAULT_CHUNK_SIZE_MB,
    ):
        self.data_type = data_type
        self.data_kind = data_kind
        self.dtype = np.dtype(dtype)
        self.run_id = run_id
        self.start = start
        self.end = end
        self.subruns = subruns
        if data is None:
            data = np.empty(0, dtype)
        self.data = data
        self.target_size_mb = target_size_mb

        if not (
            isinstance(self.start, (int, np.integer)) and isinstance(self.end, (int, np.integer))
        ):
            raise ValueError(f"Attempt to create chunk {self} with non-integer start times")
        # Convert to bona fide python integers
        self.start = int(self.start)
        self.end = int(self.end)

        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Attempt to create chunk {self} with data that isn't a numpy array")
        expected_dtype = strax.remove_titles_from_dtype(dtype)
        got_dtype = strax.remove_titles_from_dtype(dtype)
        if expected_dtype != got_dtype:
            raise ValueError(
                f"Attempt to create chunk {self} with data of {dtype}, should be {expected_dtype}"
            )
        if self.start < 0:
            raise ValueError(f"Attempt to create chunk {self} with negative start time")
        if self.start > self.end:
            raise ValueError(f"Attempt to create chunk {self} with negative length")

        if len(self.data):
            data_starts_at = self.data[0]["time"]
            # Check the last 500 samples (arbitrary number) as sanity check
            data_ends_at = strax.endtime(self.data[-500:]).max()

            if data_starts_at < self.start:
                raise ValueError(
                    f"Attempt to create chunk {self} whose data starts early at {data_starts_at}"
                )
            if data_ends_at > self.end:
                raise ValueError(
                    f"Attempt to create chunk {self} whose data ends late at {data_ends_at}"
                )

        if superrun is None:
            self.superrun = {run_id: {"start": start, "end": end}}
        else:
            if not isinstance(superrun, dict):
                raise ValueError(f"Attempt to create chunk {self} with non-dict superrun")
            if superrun == {}:
                raise ValueError(f"Attempt to create chunk {self} with empty superrun")
            self.superrun = superrun

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _t_fmt(t):
        return f"{t // int(1e9)}sec {t % int(1e9)} ns"

    def __repr__(self):
        return (
            f"({self.run_id}.{self.data_type}: "
            f"{self._t_fmt(self.start)} - {self._t_fmt(self.end)}, "
            f"{len(self)} items, " + "{0:.1f} MB/s)".format(self._mbs())
        )

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def duration(self):
        return self.end - self.start

    @property
    def is_superrun(self):
        return bool(self.subruns) and self.run_id.startswith("_")

    @property
    def first_subrun(self):
        _subrun = None
        if self.is_superrun:
            _subrun = self._get_subrun(0)
        return _subrun

    @property
    def last_subrun(self):
        _subrun = None
        if self.is_superrun:
            _subrun = self._get_subrun(-1)
        return _subrun

    def _get_subrun(self, index):
        """Returns subrun according to position in chunk."""
        subrun_id = list(self.subruns.keys())[index]
        _subrun = {
            "run_id": subrun_id,
            "start": self.subruns[subrun_id]["start"],
            "end": self.subruns[subrun_id]["end"],
        }
        return _subrun

    def _mbs(self):
        if self.duration:
            return (self.nbytes / 1e6) / (self.duration / 1e9)
        else:
            # This is strange. We have a zero duration chunk. However, this is
            # not the right place to raise an error message. Return -1 for now.
            return -1

    def split(self, t: ty.Union[int, None], allow_early_split=False):
        """Return (chunk_left, chunk_right) split at time t.

        :param t: Time at which to split the data. All data in the left chunk will have their
            (exclusive) end <= t, all data in the right chunk will have (inclusive) start >=t.
        :param allow_early_split: If False, raise CannotSplit if the requirements above cannot be
            met. If True, split at the closest possible time before t.

        """
        t = max(min(t, self.end), self.start)  # type: ignore
        if t == self.end:
            data1, data2 = self.data, self.data[:0]
        elif t == self.start:
            data1, data2 = self.data[:0], self.data
        else:
            data1, data2, t = split_array(data=self.data, t=t, allow_early_split=allow_early_split)

        common_kwargs = dict(
            run_id=self.run_id,
            dtype=self.dtype,
            data_type=self.data_type,
            data_kind=self.data_kind,
            target_size_mb=self.target_size_mb,
        )

        subruns_first_chunk, subruns_second_chunk = _split_runs_in_chunk(self.subruns, t)
        superrun_first_chunk, superrun_second_chunk = _split_runs_in_chunk(self.superrun, t)

        c1 = strax.Chunk(
            start=self.start,
            end=max(self.start, t),  # type: ignore
            data=data1,
            subruns=subruns_first_chunk,
            superrun=superrun_first_chunk,
            **common_kwargs,
        )
        c2 = strax.Chunk(
            start=max(self.start, t),  # type: ignore
            end=max(t, self.end),  # type: ignore
            data=data2,
            subruns=subruns_second_chunk,
            superrun=superrun_second_chunk,
            **common_kwargs,
        )
        return c1, c2

    @classmethod
    def merge(cls, chunks, data_type="<UNKNOWN>"):
        """Create chunk by merging columns of chunks of same data kind.

        :param chunks: Chunks to merge. None is allowed and will be ignored.
        :param data_type: data_type name of new created chunk. Set to <UNKNOWN> if not provided.

        """
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise ValueError("Need at least one chunk to merge")
        if len(chunks) == 1:
            return chunks[0]

        data_kinds = [c.data_kind for c in chunks]
        if len(set(data_kinds)) != 1:
            raise ValueError(f"Cannot merge chunks {chunks} of different data kinds: {data_kinds}")
        data_kind = data_kinds[0]

        run_ids = [c.run_id for c in chunks]
        if len(set(run_ids)) != 1:
            raise ValueError(f"Cannot merge chunks of different run_ids: {chunks}")
        run_id = run_ids[0]

        if len(set([len(c) for c in chunks])) != 1:
            raise ValueError(f"Cannot merge chunks with different number of items: {chunks}")

        tranges = [(c.start, c.end) for c in chunks]
        if len(set(tranges)) != 1:
            raise ValueError(f"Cannot merge chunks with different time ranges: {tranges}")
        start, end = tranges[0]

        data = strax.merge_arrs(
            [c.data for c in chunks],
            # Make sure dtype field order is consistent, regardless of the
            # order in which chunks are passed to merge:
            dtype=strax.merged_dtype([c.dtype for c in sorted(chunks, key=lambda x: x.data_type)]),
        )

        return cls(
            start=start,
            end=end,
            dtype=data.dtype,
            data_type=data_type,
            data_kind=data_kind,
            run_id=run_id,
            data=data,
            superrun=_merge_superrun_in_chunk(chunks, merge=True),
            target_size_mb=max([c.target_size_mb for c in chunks]),
        )

    @classmethod
    def concatenate(cls, chunks, allow_superrun=False):
        """Create chunk by concatenating chunks of same data type You can pass None's, they will be
        ignored."""
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            raise ValueError("Need at least one chunk to concatenate")
        if len(chunks) == 1:
            return chunks[0]

        data_types = [c.data_type for c in chunks]
        if len(set(data_types)) != 1:
            raise ValueError(f"Cannot concatenate chunks of different data types: {data_types}")
        data_type = data_types[0]

        run_ids = [c.run_id for c in chunks]

        if len(set(run_ids)) != 1 and not allow_superrun:
            raise ValueError(
                f"Cannot concatenate {data_type} chunks with different run ids: {run_ids}"
            )

        if len(set(run_ids)) == 1:
            run_id = run_ids[0]
            superrun = None
        else:
            run_id = None
            superrun = _merge_superrun_in_chunk(chunks)
        subruns = _merge_subruns_in_chunk(chunks)

        prev_end = 0
        for c in chunks:
            if c.start < prev_end:
                raise ValueError(
                    f"Attempt to concatenate overlapping or out-of-order chunks: {chunks} "
                )
            prev_end = c.end

        return cls(
            start=chunks[0].start,
            end=chunks[-1].end,
            dtype=chunks[0].dtype,
            data_type=data_type,
            data_kind=chunks[0].data_kind,
            run_id=run_id,
            subruns=subruns,
            superrun=superrun,
            data=np.concatenate([c.data for c in chunks]),
            target_size_mb=max([c.target_size_mb for c in chunks]),
        )


@export
def continuity_check(chunk_iter):
    """Check continuity of chunks yielded by chunk_iter as they are yielded."""
    last_end = None
    last_runid = None

    last_subrun = {"run_id": None}
    for chunk in chunk_iter:
        if chunk.run_id != last_runid:
            last_end = None
            last_subrun = {"run_id": None}

        if chunk.is_superrun:
            _subrun = chunk.first_subrun
            if _subrun["run_id"] != last_subrun["run_id"]:
                last_end = None
            else:
                last_end = last_subrun["end"]

        if last_end is not None:
            if chunk.start != last_end:
                raise ValueError(
                    f"Data is not continuous. Chunk {chunk} should have started at {last_end}"
                )
        yield chunk

        last_end = chunk.end
        last_runid = chunk.run_id
        last_subrun = chunk.last_subrun


@export
class CannotSplit(Exception):
    pass


@export
@numba.njit(cache=False, nogil=True)
def split_array(data, t, allow_early_split=False):
    """Return (data left of t, data right of t, t), or raise CannotSplit if that would split a data
    element in two.

    :param data: strax numpy data
    :param t: Time to split data
    :param allow_early_split: Instead of raising CannotSplit, split at t_split as close as possible
        before t where a split can happen. The new split time replaces t in the return value.

    """
    # Slitting an empty array is easy
    if not len(data):
        return data[:0], data[:0], t

    # Splitting off a bit of nothing from the start is easy
    # since the data is sorted by time.
    if data[0]["time"] >= t:
        return data[:0], data, t

    # Find:
    #  i_first_beyond: the first element starting after t
    #  splittable_i: nearest index left of t where we can safely split BEFORE
    latest_end_seen = -1
    splittable_i = 0
    i_first_beyond = -1
    for i, d in enumerate(data):
        if d["time"] >= latest_end_seen:
            splittable_i = i
        if d["time"] >= t:
            i_first_beyond = i
            break
        latest_end_seen = max(latest_end_seen, strax.endtime(d))
        if latest_end_seen > t:
            # Cannot split anywhere after this
            break
    else:
        if latest_end_seen <= t:
            return data, data[:0], t

    if splittable_i != i_first_beyond or latest_end_seen > t:
        if not allow_early_split:
            # Raise custom exception, make better one outside numba
            raise CannotSplit()
        t = min(data[splittable_i]["time"], t)

    return data[:splittable_i], data[splittable_i:], t


def _merge_runs_in_chunk(runs_of_chunk, merged_runs):
    """Merge subruns information during concatenation or merge."""
    if runs_of_chunk is None:
        return
    for run_id, run_start_end in runs_of_chunk.items():
        merged_runs.setdefault(run_id, [])
        merged_runs[run_id].append([run_start_end["start"], run_start_end["end"]])


def _continuity_check(merged_runs, merge=False):
    """Check continuity of runs in a superrun chunk."""
    for run_id in merged_runs.keys():
        merged_runs[run_id].sort(key=lambda x: x[0])
        if not merge:
            for i in range(1, len(merged_runs[run_id])):
                mask = merged_runs[run_id][i][0] != merged_runs[run_id][i - 1][1]
                if mask:
                    raise ValueError(
                        "Chunks are not continuous. "
                        f"Run {run_id} was split into chunks {merged_runs[run_id]}."
                    )
        else:
            for i in range(1, len(merged_runs[run_id])):
                mask = merged_runs[run_id][i][0] != merged_runs[run_id][0][0]
                mask |= merged_runs[run_id][i][1] != merged_runs[run_id][0][1]
                if mask:
                    raise ValueError(
                        "If merging, all chunks should have the same start/end time. "
                        f"But run {run_id} was split into chunks {merged_runs[run_id]}."
                    )
        merged_runs[run_id] = {
            "start": merged_runs[run_id][0][0],
            "end": merged_runs[run_id][-1][1],
        }


def _merge_subruns_in_chunk(chunks):
    """Merge list of subruns in a superrun chunk during concatenation.

    Updates also their start/ends too.

    """
    subruns = dict()
    for c_i, c in enumerate(chunks):
        _merge_runs_in_chunk(c.subruns, subruns)
    _continuity_check(subruns)
    if subruns:
        return subruns
    else:
        return None


def _merge_superrun_in_chunk(chunks, merge=False):
    """Updates superrun in a superrun chunk during concatenation."""
    superrun = dict()
    for c_i, c in enumerate(chunks):
        _merge_runs_in_chunk(c.superrun, superrun)
    _continuity_check(superrun, merge)
    return superrun


def _split_runs_in_chunk(runs_of_chunk, t):
    """Split list of runs in a superrun chunk during split.

    Updates also their start/ends too.

    """
    if runs_of_chunk is None:
        return None, None
    runs_first_chunk = {}
    runs_second_chunk = {}
    for subrun_id, subrun_start_end in runs_of_chunk.items():
        if t < subrun_start_end["start"]:
            runs_second_chunk[subrun_id] = subrun_start_end
        elif subrun_start_end["start"] <= t < subrun_start_end["end"]:
            runs_first_chunk[subrun_id] = {"start": subrun_start_end["start"], "end": int(t)}
            runs_second_chunk[subrun_id] = {"start": int(t), "end": subrun_start_end["end"]}
        elif subrun_start_end["end"] <= t:
            runs_first_chunk[subrun_id] = subrun_start_end
    # Make sure that either dictionary with content or None is assigned to Chunk
    if runs_first_chunk == {}:
        runs_first_chunk = None
    if runs_second_chunk == {}:
        runs_second_chunk = None
    return runs_first_chunk, runs_second_chunk


@export
class Rechunker:
    """Helper class for rechunking.

    Send in chunks via receive, which returns either None (no chunk to send) or a chunk to send.

    Don't forget a final call to .flush() to get any final data out!

    """

    def __init__(self, rechunk=False, run_id=None):
        self.rechunk = rechunk
        self.is_superrun = run_id and run_id.startswith("_")
        self.run_id = run_id

        self.cache = None

    def receive(self, chunk) -> list:
        """Receive a chunk, return list of chunks to send out after merging and splitting."""
        if not self.rechunk:
            # We aren't rechunking
            return [chunk]

        if self.cache is not None:
            # We have an old chunk, so we need to concatenate
            # We do not expect after concatenation that the chunk will be very large because
            # the self.cache is already after splitting according to the target size
            chunk = strax.Chunk.concatenate([self.cache, chunk])

        target_size_b = chunk.target_size_mb * 1e6

        # Get the split indices according to the allowed minimum gaps
        # between data and the target size of chunk
        split_indices = self.get_splits(chunk.data, target_size_b, DEFAULT_CHUNK_SPLIT_NS)
        # Split the cache into chunks and return list of chunks
        chunks = []
        for index in split_indices:
            _chunk, chunk = chunk.split(
                t=chunk.data["time"][index] - int(DEFAULT_CHUNK_SPLIT_NS // 2),
                allow_early_split=False,
            )
            chunks.append(_chunk)
        self.cache = chunk
        return chunks

    def flush(self) -> list:
        """Flush the cache and return the remaining chunk in a list."""
        if self.cache is None:
            return []
        else:
            result = self.cache
            self.cache = None
            return [result]

    @staticmethod
    def get_splits(data, target_size, min_gap=DEFAULT_CHUNK_SPLIT_NS):
        """Get indices where to split the data into chunks of approximately target_size."""
        assumed_i = int(target_size // data.itemsize)
        gap_indices = np.argwhere(strax.diff(data) > min_gap).flatten() + 1
        split_indices = [0]
        if len(gap_indices) != 0:
            while split_indices[-1] + assumed_i < gap_indices[-1]:
                split_indices.append(
                    gap_indices[np.abs(gap_indices - assumed_i - split_indices[-1]).argmin()]
                )
        split_indices = np.diff(split_indices)
        return split_indices
