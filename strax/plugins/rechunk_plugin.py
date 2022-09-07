import typing
import numpy as np
import strax
from .plugin import Plugin
export, __all__ = strax.exporter()

@export
class RechunkerPlugin(Plugin):
    """
   This is buffering plugin. It keeps a state and as such does not allow
   for parallization and other nice strax features (similar to the
   OverLapWindowPlugin). It's intended to be used for data that is
   sparse in times and is hard to fit into the chunking paradigm.
   Examples are:
    - handing logic signals, that can span minutes and are NOT sorted in
      endtime (assumed by OverlapWindowPlugin), and requite a lot of
      overhead if all chunks were buffered in an OverlapWindowPlugin.
    - Simulation plugins where one might realize that there was actually
      a bit of information that needs to be added to the result of the
      plugin somewhat later (the input is not sorted by time).

   Works similar to the OverLapWindowPlugin.

   Required class functions:
     - send_input_after, should
     - min_gap_in_chunk
   """
    parallel = False
    max_buffer_mb = 1_000
    _NO_SPLIT_FOUND = -42
    rechunk_on_save = True

    def send_input_after(self) -> int:
        """After how long buffering can we start sending the output"""
        raise NotImplementedError

    def min_gap_in_chunk(self) -> int:
        """How long does the gap need to be minimally to the next chunk"""
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.cached_input = {}

    def iter(self, iters, executor=None) -> typing.Union[
        strax.Chunk,
        typing.Dict[str, strax.Chunk]
    ]:
        yield from super().iter(iters, executor=executor)

        # Yield final results
        if self.cached_input:
            final_chunk = self.get_final()
            yield final_chunk

    def _input_is_old_enough(self) -> bool:
        """
        Did we wait long enough to try start sending data,
        check is not run on the last bit of data on the final iter
        """
        for data_type, new_input in self.cached_input.items():
            t0 = self.cached_input.get(data_type, new_input).start
            t1 = new_input.end
            # We need to have to start sending results
            if t1 - t0 < self.send_input_after() + self.min_gap_in_chunk():
                return False
        return True

    def _input_to_cache(self, kwargs: typing.Dict[str, strax.Chunk]) -> None:
        """Push new input to the cache"""
        for data_kind, chunk in kwargs.items():
            if data_kind in self.cached_input:
                self.cached_input[data_kind] = strax.Chunk.concatenate(
                    [self.cached_input[data_kind], chunk])
            else:
                self.cached_input[data_kind] = chunk

    def _empty_chunk_for_kwargs(self, kwargs) -> typing.Union[
        strax.Chunk,
        typing.Dict[str, strax.Chunk]
    ]:
        """Return an empty dict for each of the datatypes produced"""
        # Just get one chunk to copy the start of
        input_chunk = kwargs[strax.to_str_tuple(self.depends_on)[0]]
        print(input_chunk)
        provides = strax.to_str_tuple(self.provides)
        empty_chunks = {
            data_type:
                strax.Chunk(
                    # NB! The return chunk is an empty 0-length chunk!
                    start=input_chunk.start,
                    end=input_chunk.start,
                    data=None,
                    # Just copy these properties
                    data_type=data_type,
                    data_kind=self.data_kind_for(data_type),
                    dtype=self.dtype_for(data_type),
                    run_id=input_chunk.run_id,
                    subruns=input_chunk.subruns,
                )
            for data_type in provides
        }
        if self.multi_output:
            return empty_chunks
        # Only one provide, so return one bare chunk
        return empty_chunks[provides[0]]

    _prev_empty_chunk = []

    # could be optimized a bit / use numba
    def _find_split_in_input(self) -> int:
        """
        Find the latest time where the gap to the next start time is
        larger than the requested gap
        """
        if not self.cached_input or any(not len(c) for c in self.cached_input.values()):
            return self._NO_SPLIT_FOUND

        all_starts = np.concatenate([c.data['time'] for c in self.cached_input.values()])
        all_ends = np.concatenate([strax.endtime(c.data) for c in self.cached_input.values()])
        old_enough = all_ends < all_ends[-1] - self.send_input_after()

        sort = np.argsort(all_starts[old_enough])
        starts = all_starts[old_enough][sort]
        ends = all_ends[old_enough][sort]
        del all_ends, all_starts, old_enough
        gap = starts[1:] - ends[:-1]
        dt_suffices = gap > self.min_gap_in_chunk()
        if np.any(dt_suffices):
            last_stop = ends[-1:][dt_suffices[-1]]
            return last_stop
        return self._NO_SPLIT_FOUND

    def get_final(self) -> typing.Union[
        strax.Chunk,
        typing.Dict[str, strax.Chunk]
    ]:
        """
        Compute the results of whatever input is left in the input buffer and clean it
        """
        result = super().do_compute(**self.cached_input)
        del self.cached_input
        return result

    def do_compute(self, chunk_i=None, **kwargs) -> typing.Union[
        strax.Chunk,
        typing.Dict[str, strax.Chunk]
    ]:
        self._input_to_cache(kwargs)

        if not self._input_is_old_enough():
            self._prev_empty_chunk.append(self._empty_chunk_for_kwargs(kwargs))
            return self._prev_empty_chunk[0]

        # Should be plenty space after this point, but let's assume
        # someone (or me) can make an off by one error
        last_split_t = self._find_split_in_input() + 1
        if last_split_t == self._NO_SPLIT_FOUND:
            return self._empty_chunk_for_kwargs(kwargs)
        chunks_for_compute = {}
        for data_type, chunk in self.cached_input.items():
            early_chunk_bit, late_chunk_bit = chunk.split(t=last_split_t,
                                                          allow_early_split=False)
            chunks_for_compute[data_type] = early_chunk_bit
            self.cached_input[data_type] = late_chunk_bit

        # Compute new results
        result = super().do_compute(chunk_i=chunk_i, **kwargs)
        return result

