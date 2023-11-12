import strax
from .plugin import Plugin

export, __all__ = strax.exporter()


@export
class OverlapWindowPlugin(Plugin):
    """Plugin whose computation depends on having its inputs extend a certain window on both sides.

    Current implementation assumes:
    - All inputs are sorted by *endtime*. Since everything in strax is sorted
    by time, this only works for disjoint intervals such as peaks or events,
    but NOT records!
    - You must read time info for your data kind, or create a new data kind.

    """

    parallel = False

    def __init__(self):
        super().__init__()
        self.cached_input = {}
        self.cached_results = None
        self.sent_until = 0
        # This guy can have a logger, it's not parallelized anyway

    def get_window_size(self):
        """Return the required window size in nanoseconds."""
        raise NotImplementedError

    def iter(self, iters, executor=None):
        yield from super().iter(iters, executor=executor)

        # Yield final results, kept at bay in fear of a new chunk
        if self.cached_results is not None:
            yield self.cached_results

    def do_compute(self, chunk_i=None, **kwargs):
        if not len(kwargs):
            raise RuntimeError("OverlapWindowPlugin must have a dependency")

        # Add cached inputs to compute arguments
        for data_kind, chunk in kwargs.items():
            if len(self.cached_input):
                kwargs[data_kind] = strax.Chunk.concatenate([self.cached_input[data_kind], chunk])

        # Compute new results
        result = super().do_compute(chunk_i=chunk_i, **kwargs)

        # Throw away results we already sent out
        _, result = result.split(t=self.sent_until, allow_early_split=False)

        # When does this batch of inputs end?
        ends = [c.end for c in kwargs.values()]
        if not len(set(ends)) == 1:
            raise RuntimeError(f"OverlapWindowPlugin got incongruent inputs: {kwargs}")
        end = ends[0]

        # When can we no longer trust our results?
        # Take slightly larger windows for safety: it is very easy for me
        # (or the user) to have made an off-by-one error
        invalid_beyond = int(end - self.get_window_size() - 1)

        # Prepare to send out valid results, cache the rest
        # Do not modify result anymore after this
        # Note result.end <= invalid_beyond, with equality if there are
        # no overlaps
        result, self.cached_results = result.split(t=invalid_beyond, allow_early_split=True)
        self.sent_until = result.end

        # Cache a necessary amount of input for next time
        # Again, take a bit of overkill for good measure
        cache_inputs_beyond = int(self.sent_until - 2 * self.get_window_size() - 1)

        # Cache inputs, make sure that the chunks start at the same time to
        # prevent issues in input buffers later on
        prev_split = cache_inputs_beyond
        max_trials = 10
        for try_counter in range(max_trials):
            for data_kind, chunk in kwargs.items():
                _, self.cached_input[data_kind] = chunk.split(t=prev_split, allow_early_split=True)
                prev_split = self.cached_input[data_kind].start

            unique_starts = set([c.start for c in self.cached_input.values()])
            chunk_starts_are_equal = len(unique_starts) == 1
            if chunk_starts_are_equal:
                self.log.debug(
                    f"Success after {try_counter}. Extra time = {cache_inputs_beyond-prev_split} ns"
                )
                break
            else:
                self.log.debug(
                    "Inconsistent start times of the cashed chunks after"
                    f" {try_counter}/{max_trials} passes.\nChunks {self.cached_input}"
                )
        else:
            raise ValueError(
                f"Buffer start time inconsistency cannot be resolved after {max_trials} tries"
            )
        return result
