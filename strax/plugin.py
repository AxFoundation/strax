"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from enum import IntEnum
from functools import partial

import numpy as np

import strax
import strax.chunk_arrays as ca
export, __all__ = strax.exporter()


@export
class SaveWhen(IntEnum):
    """Plugin's preference for having it's data saved"""
    NEVER = 0         # Throw an error if the user lists it
    EXPLICIT = 1      # Save ONLY if the user lists it explicitly
    TARGET = 2        # Save if the user asks for it as a final target
    ALWAYS = 3        # Save even if the user does not list it


@export
class Plugin:
    """Plugin containing strax computation

    You should NOT instantiate plugins directly.
    """
    __version__ = '0.0.0'
    data_kind: str
    depends_on: tuple
    provides: str
    dependency_kinds: dict
    dependency_dtypes: dict
    compressor = 'blosc'
    n_per_iter = None
    rechunk = True

    save_when = SaveWhen.TARGET
    parallel = False    # If True, compute() work is submitted to pool

    def startup(self):
        """Hook if plugin wants to do something after initialization."""
        pass

    def infer_dtype(self):
        """Return dtype of computed data;
        used only if no dtype attribute defined"""
        raise NotImplementedError

    def version(self, run_id=None):
        """Return version number applicable to the run_id.
        Most plugins just have a single version (in .__version__)
        but some may be at different versions for different runs
        (e.g. time-dependent corrections).
        """
        return self.__version__

    def lineage(self, run_id):
        # TODO: Implement this
        return None

    def metadata(self, run_id):
        """Metadata to save along with produced data"""
        return dict(
            run_id=run_id,
            data_kind=self.data_kind,
            compressor=self.compressor,
            dtype=self.dtype,
            version=self.version(run_id),
            lineage=self.lineage(run_id)
        )

    def dependencies_by_kind(self, require_time=True):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True (default), one dependency of each kind
        must provide time information. It will be put first in the list.
        """
        deps_by_kind = dict()
        key_deps = []
        for d in self.depends_on:
            k = self.dependency_kinds[d]
            deps_by_kind.setdefault(k, [])

            # If this has time information, put it first in the list
            if (require_time
                    and 'time' in self.dependency_dtypes[d].names):
                key_deps.append(d)
                deps_by_kind[k].insert(0, d)
            else:
                deps_by_kind[k].append(d)

        if require_time:
            for k, d in deps_by_kind.items():
                if not d[0] in key_deps:
                    raise ValueError(f"No dependency of data kind {k} "
                                     "has time information!")

        return deps_by_kind

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results
        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to.
            If None, will compute inside the plugin's thread.
        """
        deps_by_kind = self.dependencies_by_kind()

        if self.n_per_iter is not None:
            # Apply additional flow control
            for kind, deps in deps_by_kind.items():
                d = deps[0]
                iters[d] = ca.fixed_length_chunks(iters[d], n=self.n_per_iter)
                break

        if len(deps_by_kind) > 1:
            # Sync the iterators that provide time info for each data kind
            # (first in deps_by_kind lists) by endtime
            iters.update(ca.sync_iters(
                partial(ca.same_stop, func=strax.endtime),
                {d[0]: iters[d[0]]
                 for d in deps_by_kind.values()}))

        # Sync the iterators of each data_kind to provide same-length chunks
        for deps in deps_by_kind.values():
            if len(deps) > 1:
                iters.update(ca.sync_iters(
                    ca.same_length,
                    {d: iters[d] for d in deps}))

        while True:
            try:
                compute_kwargs = {d: next(iters[d])
                                  for d in self.depends_on}
            except StopIteration:
                return
            if self.parallel and executor is not None:
                yield executor.submit(self.compute, **compute_kwargs)
            else:
                yield self.compute(**compute_kwargs)

    @staticmethod
    def compute(**kwargs):
        raise NotImplementedError


##
# Special plugins
##

@export
class LoopPlugin(Plugin):
    """Plugin that disguises multi-kind data-iteration by an event loop
    """

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for LoopPlugin')
        super().__init__()

    def compute(self, **kwargs):
        # If not otherwise specified, data kind to loop over
        # is that of the first dependency (e.g. events)
        if hasattr(self, 'loop_over'):
            loop_over = self.loop_over
        else:
            loop_over = self.dependency_kinds[self.depends_on[0]]

        # Merge data of each data kind
        deps_by_kind = self.dependencies_by_kind()
        things_by_kind = {
            k: strax.merge_arrs([kwargs[d] for d in deps])
            for k, deps in deps_by_kind.items()
        }

        # Group into lists of things (e.g. peaks)
        # contained in the base things (e.g. events)
        base = things_by_kind[loop_over]
        for k, things in things_by_kind.items():
            if k != loop_over:
                r = strax.split_by_containment(things, base)
                if len(r) != len(base):
                    print(f"Last base: {base[-1]['time']}-{strax.endtime(base[-1])}")
                    print(f"Last ting: {things[-1]['time']}-{strax.endtime(things[-1])}")
                    raise RuntimeError(f"Split {k} into {len(r)}, should be {len(base)}!")
                things_by_kind[k] = r

        results = np.zeros(len(base), dtype=self.dtype)
        for i in range(len(base)):
            r = self.compute_loop(base[i],
                                  **{k: things_by_kind[k][i]
                                     for k in deps_by_kind
                                     if k != loop_over})

            # Convert from dict to array row:
            for k, v in r.items():
                results[i][k] = v

        return results

    def compute_loop(self, *args, **kwargs):
        raise NotImplementedError


@export
class MergePlugin(Plugin):
    """Plugin that merges data from its dependencies
    """
    save_when = SaveWhen.EXPLICIT

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for MergePlugin')

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergePlugins can only merge data of the same "
                             "kind, but got multiple kinds: "
                             + str(deps_by_kind))

        return sum([strax.unpack_dtype(self.dependency_dtypes[d])
                    for d in self.depends_on], [])

    def compute(self, **kwargs):
        return strax.merge_arrs(list(kwargs.values()))


@export
class PlaceholderPlugin(Plugin):
    """Plugin that throws NotImplementedError when asked to compute anything"""
    depends_on = tuple()
    save_when = SaveWhen.NEVER

    def compute(self):
        raise NotImplementedError("No plugin registered that "
                                  f"provides {self.provides}")
