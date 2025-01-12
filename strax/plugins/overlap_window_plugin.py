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
    max_trials = 10

    def __init__(self):
        super().__init__()
        self.cached_input = {}
        if self.multi_output:
            self.cached_results = {}
        else:
            self.cached_results = None
        self.sent_until = 0
        if self.clean_chunk_after_compute:
            raise ValueError(
                "OverlapWindowPlugin cannot clean chunks after compute because you need them later."
            )
        # This guy can have a logger, it's not parallelized anyway

    def get_window_size(self):
        """Return the required window size in nanoseconds."""
        raise NotImplementedError

    def iter(self, iters, executor=None):
        yield from super().iter(iters, executor=executor)

        # Yield final results, kept at bay in fear of a new chunk
        yield self.cached_results

    def do_compute(self, chunk_i=None, **kwargs):
        if not len(kwargs):
            raise RuntimeError("OverlapWindowPlugin must have a dependency")

        # Add cached inputs to compute arguments
        for data_kind, chunk in kwargs.items():
            if len(self.cached_input):
                kwargs[data_kind] = strax.Chunk.concatenate(
                    [self.cached_input[data_kind], chunk], self.allow_superrun
                )

        # When does this batch of inputs end?
        ends = [c.end for c in kwargs.values()]
        if not len(set(ends)) == 1:
            raise RuntimeError(f"OverlapWindowPlugin got incongruent inputs: {kwargs}")
        end = ends[0]

        # When can we no longer trust our results?
        # Take slightly larger windows for safety: it is very easy for me
        # (or the user) to have made an off-by-one error
        invalid_beyond = int(end - 2 * self.get_window_size() - 1)

        # Compute new results
        result = super().do_compute(chunk_i=chunk_i, **kwargs)

        # Throw away results we already sent out
        # no error here though allow_early_split=False,
        # because result.split(t=invalid_beyond, allow_early_split=True) tunes the
        # sent_until to be not overlapping with result and
        # sent_until <= invalid_beyond
        if self.multi_output:
            # when multi_output=True, the result is a dict
            for data_type in result:
                result[data_type] = result[data_type].split(
                    t=self.sent_until, allow_early_split=False
                )[1]
        else:
            result = result.split(t=self.sent_until, allow_early_split=False)[1]

        # Prepare to send out valid results, cache the rest
        # Do not modify result anymore after these lines
        # Note result.end <= invalid_beyond, with equality if there are no overlaps
        if self.multi_output:
            prev_split = self.cache_beyond(result, invalid_beyond, self.cached_results)
            for data_type in result:
                result[data_type], self.cached_results[data_type] = result[data_type].split(
                    t=prev_split, allow_early_split=True
                )
            if len(set([c.start for c in self.cached_results.values()])) != 1:
                raise ValueError("Output start time inconsistency has not been resolved?")
            self.sent_until = prev_split
        else:
            result, self.cached_results = result.split(t=invalid_beyond, allow_early_split=True)
            self.sent_until = self.cached_results.start

        # Cache a necessary amount of input for next time
        # Again, take a bit of overkill for good measure
        # cache_inputs_beyond is smaller than sent_until
        cache_inputs_beyond = int(self.sent_until - 2 * self.get_window_size() - 1)

        # Cache inputs, make sure that the chunks start at the same time to
        # prevent issues in input buffers later on
        self.cache_beyond(kwargs, cache_inputs_beyond, self.cached_input)
        return result

    def cache_beyond(self, io, prev_split, cached):
        original_prev_split = prev_split
        for try_counter in range(self.max_trials):
            for data, chunk in io.items():
                # data here can not either data_kind or data_type
                # do not temporarily modify result here because it will be used later
                # keep its original value!
                cached[data] = chunk.split(t=prev_split, allow_early_split=True)[1]
                prev_split = cached[data].start
            unique_starts = set([c.start for c in cached.values()])
            if len(unique_starts) == 1:
                self.log.debug(
                    f"Success after {try_counter}. "
                    f"Extra time is {original_prev_split - prev_split} ns"
                )
                break
            else:
                self.log.debug(
                    "Inconsistent start times of the cashed chunks {io} after"
                    f" {try_counter}/{self.max_trials} passes."
                )
        else:
            raise ValueError(
                f"Buffer start time inconsistency cannot be resolved after {self.max_trials} tries"
            )
        return prev_split
