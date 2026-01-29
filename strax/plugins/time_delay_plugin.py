"""Plugin base class for algorithms that add time delays to output."""

import numpy as np
import strax
from .plugin import Plugin

export, __all__ = strax.exporter()


@export
class TimeDelayPlugin(Plugin):
    """Plugin base class for algorithms that add time delays to output.

    Use this when your algorithm shifts output timestamps forward in time,
    potentially beyond input chunk boundaries. Handles variable delays with
    known maximum, re-sorting, buffering across chunk boundaries, and
    multi-output plugins.

    Subclasses must implement:
        compute_with_delay(**kwargs): Return delayed output data (arrays, not Chunks)

    For multi-output plugins, compute_with_delay should return a dict
    mapping data_type names to numpy arrays.

    """

    parallel = False

    def __init__(self):
        super().__init__()
        self.output_buffer = {}
        self.last_output_end = 0
        self.first_output = True
        self._cached_superrun = None
        self._cached_subruns = None
        self._min_buffered_time = float("inf")

    def compute_with_delay(self, **kwargs):
        """Compute output data with time delays already applied.

        Input arrays are numpy arrays (not Chunks). Output arrays do NOT need to be sorted. For
        multi-output, return a dict mapping data_type to arrays.

        """
        raise NotImplementedError("Subclasses must implement compute_with_delay()")

    def iter(self, iters, executor=None):
        """Override iter to flush buffer at end of processing."""
        yield from super().iter(iters, executor=executor)
        final_result = self._flush_buffers()
        if final_result is not None:
            yield final_result

    def _flush_buffers(self):
        """Flush all remaining data from buffers."""
        has_data = any(len(self.output_buffer.get(dt, [])) > 0 for dt in self.provides)
        if not has_data:
            return None

        # Sort buffers and compute chunk_end
        chunk_end = self.last_output_end
        for data_type in self.provides:
            buf = self.output_buffer.get(data_type)
            if buf is not None and len(buf) > 0:
                buf.sort(order="time")
                data_end = int(strax.endtime(buf).max())
                chunk_end = max(chunk_end, data_end)

        # Build result dict
        result = {}
        for data_type in self.provides:
            buf = self.output_buffer.get(data_type, np.empty(0, self.dtype_for(data_type)))
            result[data_type] = self._make_chunk(
                data=buf,
                data_type=data_type,
                start=self.last_output_end,
                end=chunk_end,
            )

        result = self.superrun_transformation(result, self._cached_superrun, self._cached_subruns)
        self.output_buffer = {}

        return self._unwrap_result(result)

    def do_compute(self, chunk_i=None, **kwargs):
        """Process input, buffer output, return safe portion."""
        if not kwargs:
            raise RuntimeError("TimeDelayPlugin must have dependencies")
        first_chunk = next(iter(kwargs.values()))
        input_end = first_chunk.end

        self._cached_superrun = self._check_subruns_uniqueness(
            kwargs, {k: v.superrun for k, v in kwargs.items()}
        )
        self._cached_subruns = self._check_subruns_uniqueness(
            kwargs, {k: v.subruns for k, v in kwargs.items()}
        )

        input_data = {k: v.data for k, v in kwargs.items()}
        new_output = self.compute_with_delay(**input_data)

        self._add_to_buffers(new_output)

        return self._process_output(safe_boundary=input_end)

    def _unwrap_result(self, result):
        """Unwrap result dict to single Chunk for single-output plugins."""
        if self.multi_output:
            return result
        return result[self.provides[0]]

    def _add_to_buffers(self, new_output):
        """Add new output to buffers."""
        # Normalize output to dict format
        if self.multi_output:
            if not isinstance(new_output, dict):
                raise ValueError(
                    f"{self.__class__.__name__} is multi-output, "
                    "compute_with_delay must return a dict"
                )
            output_dict = new_output
        else:
            if isinstance(new_output, dict):
                raise ValueError(
                    f"{self.__class__.__name__} is single-output, "
                    "compute_with_delay should not return a dict"
                )
            output_dict = {self.provides[0]: new_output}

        for data_type in self.provides:
            arr = output_dict.get(data_type, np.empty(0, self.dtype_for(data_type)))
            if not isinstance(arr, np.ndarray):
                arr = strax.dict_to_rec(arr, dtype=self.dtype_for(data_type))

            if data_type not in self.output_buffer:
                self.output_buffer[data_type] = arr
            elif len(arr) > 0:
                self.output_buffer[data_type] = np.concatenate([self.output_buffer[data_type], arr])

    def _process_output(self, safe_boundary):
        """Process buffers and return safe portion."""
        # Sort all buffers
        for data_type in self.provides:
            buf = self.output_buffer.get(data_type)
            if buf is not None and len(buf) > 0:
                buf.sort(order="time")

        # Split buffers into safe and remaining portions
        safe_data_dict = {}
        for data_type in self.provides:
            buf = self.output_buffer.get(data_type, np.empty(0, self.dtype_for(data_type)))
            safe_data, remaining = self._split_buffer(buf, safe_boundary)
            self.output_buffer[data_type] = remaining
            safe_data_dict[data_type] = safe_data

        # Update minimum buffered time
        min_time = float("inf")
        for buf in self.output_buffer.values():
            if buf is not None and len(buf) > 0:
                min_time = min(min_time, buf["time"].min())
        self._min_buffered_time = min_time

        # Compute unified chunk boundaries across all data types
        chunk_start = None
        chunk_end = None
        for data_type in self.provides:
            safe_data = safe_data_dict[data_type]
            dt_start, dt_end = self._get_chunk_boundaries(safe_data, safe_boundary)
            if chunk_start is None:
                chunk_start = dt_start
                chunk_end = dt_end
            else:
                chunk_start = min(chunk_start, dt_start)
                chunk_end = max(chunk_end, dt_end)

        # Build result dict
        result = {}
        for data_type in self.provides:
            result[data_type] = self._make_chunk(
                data=safe_data_dict[data_type],
                data_type=data_type,
                start=chunk_start,
                end=chunk_end,
            )

        self.last_output_end = chunk_end
        self.first_output = False

        result = self.superrun_transformation(result, self._cached_superrun, self._cached_subruns)

        return self._unwrap_result(result)

    def _split_buffer(self, buf, safe_boundary):
        """Split buffer into safe portion (endtime <= boundary) and remainder."""
        if len(buf) == 0:
            empty = np.empty(0, buf.dtype)
            return empty, empty

        endtimes = strax.endtime(buf)
        safe_mask = endtimes <= safe_boundary

        safe_data = buf[safe_mask].copy()
        remaining = buf[~safe_mask].copy()

        return safe_data, remaining

    def _get_chunk_boundaries(self, safe_data, safe_boundary):
        """Determine chunk start/end ensuring buffered data fits in next chunk."""
        if self.first_output:
            if len(safe_data) > 0:
                chunk_start = int(safe_data[0]["time"])
            else:
                chunk_start = 0
        else:
            chunk_start = self.last_output_end

        if len(safe_data) > 0:
            data_end = int(strax.endtime(safe_data).max())
            chunk_end = max(data_end, safe_boundary)
        else:
            chunk_end = safe_boundary

        # Don't advance chunk_end past minimum buffered time
        if self._min_buffered_time < float("inf"):
            chunk_end = min(chunk_end, int(self._min_buffered_time))

        chunk_end = max(chunk_start, chunk_end)

        return chunk_start, chunk_end

    def _make_chunk(self, data, data_type, start, end):
        """Create a strax Chunk with proper metadata."""
        return strax.Chunk(
            start=start,
            end=end,
            data=data,
            data_type=data_type,
            data_kind=self.data_kind_for(data_type),
            dtype=self.dtype_for(data_type),
            run_id=self._run_id,
            target_size_mb=self.chunk_target_size_mb,
        )
