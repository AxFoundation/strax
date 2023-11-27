import numpy as np
import strax
from .plugin import Plugin
from immutabledict import immutabledict
from warnings import warn

export, __all__ = strax.exporter()


@export
class LoopPlugin(Plugin):
    """Plugin that disguises multi-kind data-iteration by an event loop."""

    # time_selection: Kind of time selection to apply:
    # - touching: select things that (partially) overlap with the range.
    # NB! Use this option with care since if e.g. two events are
    # adjacent, touching windows might return ambiguous results as peaks
    # may be touching both events.
    # The number of samples to be desired to overlapped can be set by
    # self.touching_window. Otherwise 0 is assumed (see strax.touching_windows)
    # - fully_contained: (default) select things fully contained in the range
    time_selection = "fully_contained"

    def compute(self, **kwargs):
        # If not otherwise specified, data kind to loop over
        # is that of the first dependency (e.g. events)
        # Can't be in __init__: deps not initialized then
        if hasattr(self, "loop_over"):
            loop_over = self.loop_over
        else:
            loop_over = self.deps[self.depends_on[0]].data_kind
        if not isinstance(loop_over, str):
            raise TypeError('Please add "loop_over = <base>" to your plugin definition')

        # Group into lists of things (e.g. peaks)
        # contained in the base things (e.g. events)
        base = kwargs[loop_over]
        if len(base) > 1:
            assert np.all(base[1:]["time"] >= strax.endtime(base[:-1])), f"{base}s overlap"

        for k, things in kwargs.items():
            # Check for sorting
            difs = np.diff(things["time"])
            if difs.min(initial=0) < 0:
                i_bad = np.argmin(difs)
                examples = things[i_bad - 1 : i_bad + 3]
                t0 = examples["time"].min()
                raise ValueError(
                    f"Expected {k} to be sorted, but found "
                    + str([(x["time"] - t0, strax.endtime(x) - t0) for x in examples])
                )

            if k != loop_over:
                if self.time_selection == "fully_contained":
                    r = strax.split_by_containment(things, base)
                elif self.time_selection == "touching":
                    # Experimental feature that should be handled with care:
                    # github.com/AxFoundation/strax/pull/424
                    warn(
                        f"{self.__class__.__name__} has a touching time "
                        "selection. This may lead to ambiguous results as two "
                        f"{loop_over}'s may contain the same {k}, thereby a "
                        f"given {k} can be included multiple times."
                    )
                    window = 0
                    if hasattr(self, "touching_window"):
                        window = self.touching_window
                    r = strax.split_touching_windows(things, base, window=window)
                else:
                    raise RuntimeError("Unknown time_selection")
                if len(r) != len(base):
                    raise RuntimeError(f"Split {k} into {len(r)}, should be {len(base)}!")
                kwargs[k] = r

        if self.multi_output:
            # This is the a-typical case. Most of the time you just have
            # one output. Just doing the same as below but this time we
            # need to create a dict for the outputs.
            # NB: both outputs will need to have the same length as the
            # base!
            results = {k: np.zeros(len(base), dtype=self.dtype[k]) for k in self.provides}
            deps_by_kind = self.dependencies_by_kind()

            for i, base_chunk in enumerate(base):
                res = self.compute_loop(
                    base_chunk, **{k: kwargs[k][i] for k in deps_by_kind if k != loop_over}
                )
                if not isinstance(res, (dict, immutabledict)):
                    raise AttributeError("Please provide result in compute loop as dict")
                # Convert from dict to array row:
                for provides, r in res.items():
                    for k, v in r.items():
                        if np.shape(v) != np.shape(results[provides][i][k]):
                            # Make sure that the buffer length as
                            # defined by the base matches the output of
                            # the compute argument.
                            raise ValueError(
                                f"{provides} returned an improper length array "
                                f"that is not equal to the {loop_over} "
                                "data-kind! Are you sure a LoopPlugin is the "
                                "right Plugin for your application?"
                            )
                        results[provides][i][k] = v
        else:
            # Normally you end up here were we are going to loop over
            # base and add the results to the right format.
            results = np.zeros(len(base), dtype=self.dtype)
            deps_by_kind = self.dependencies_by_kind()

            for i, base_chunk in enumerate(base):
                r = self.compute_loop(
                    base_chunk, **{k: kwargs[k][i] for k in deps_by_kind if k != loop_over}
                )
                if not isinstance(r, (dict, immutabledict)):
                    raise AttributeError("Please provide result in compute loop as dict")
                # Convert from dict to array row:
                for k, v in r.items():
                    results[i][k] = v
        return results

    def compute_loop(self, *args, **kwargs):
        raise NotImplementedError
