"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from concurrent.futures import wait
from enum import IntEnum
import itertools
from functools import partial
import typing

import numpy as np

import strax
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
    Do NOT add unpickleable things (e.g. loggers) as attributes.
    """
    __version__ = '0.0.0'
    data_kind: str
    depends_on: tuple
    provides: str

    compressor = 'blosc'
    n_per_iter = None       # TODO: think about this
    rechunk = True
    save_when = SaveWhen.ALWAYS
    parallel = False    # If True, compute() work is submitted to pool
    save_meta_only = False

    # These are set on plugin initialization, which is done in the core
    run_id: str
    config: typing.Dict
    deps: typing.List       # Dictionary of dependency plugin instances
    takes_config = dict()       # Config options
    compute_takes_chunk_i = False  # Autoinferred, no need to set yourself

    def __init__(self):
        self.pre_compute = []
        self.post_compute = []
        self.on_close = []

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

    def metadata(self, run_id):
        """Metadata to save along with produced data"""
        return dict(
            run_id=run_id,
            data_type=self.provides,
            data_kind=self.data_kind,
            dtype=self.dtype,
            compressor=self.compressor,
            lineage=self.lineage)

    def dependencies_by_kind(self, require_time=True):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True (default), one dependency of each kind
        must provide time information. It will be put first in the list.
        """
        deps_by_kind = dict()
        key_deps = []
        for d in self.depends_on:
            k = self.deps[d].data_kind
            deps_by_kind.setdefault(k, [])

            # If this has time information, put it first in the list
            if (require_time
                    and 'time' in self.deps[d].dtype.names):
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
                iters[d] = strax.fixed_length_chunks(iters[d],
                                                     n=self.n_per_iter)
                break

        if len(deps_by_kind) > 1:
            # Sync the iterators that provide time info for each data kind
            # (first in deps_by_kind lists) by endtime
            iters.update(strax.sync_iters(
                partial(strax.same_stop, func=strax.endtime),
                {d[0]: iters[d[0]]
                 for d in deps_by_kind.values()}))

        # Sync the iterators of each data_kind to provide same-length chunks
        for deps in deps_by_kind.values():
            if len(deps) > 1:
                iters.update(strax.sync_iters(
                    strax.same_length,
                    {d: iters[d] for d in deps}))

        pending = []
        for chunk_i in itertools.count():
            try:
                if not self.check_next_ready_or_done(chunk_i):
                    raise StopIteration
                compute_kwargs = {d: next(iters[d])
                                  for d in self.depends_on}
            except StopIteration:
                self.close(wait_for=tuple(pending))
                return
            except Exception:
                self.close(wait_for=tuple(pending))
                raise

            if self.parallel and executor is not None:
                new_f = executor.submit(self.do_compute,
                                        chunk_i=chunk_i,
                                        **compute_kwargs)
                pending = [f for f in pending + [new_f]
                           if not f.done()]
                yield new_f
            else:
                yield self.do_compute(chunk_i=chunk_i, **compute_kwargs)

    def do_compute(self, *args, chunk_i=None, **kwargs):
        for i, x in enumerate(args):
            kwargs[self.depends_on[i]] = x

        for p in self.pre_compute:
            r = p(**kwargs, chunk_i=chunk_i)
            if r is not None:
                kwargs = r

        if self.compute_takes_chunk_i:
            result = self.compute(**kwargs, chunk_i=chunk_i)
        else:
            result = self.compute(**kwargs)

        for p in self.post_compute:
            r = p(result, chunk_i=chunk_i)
            if r is not None:
                result = r

        return result

    def check_next_ready_or_done(self, chunk_i):
        return True

    def close(self, wait_for=tuple(), timeout=30):
        wait(wait_for, timeout=timeout)
        for x in self.on_close:
            x()

    def compute(self, **kwargs):
        raise NotImplementedError


##
# Special plugins
##


@export
class MergePlugin(Plugin):

    def __init__(self):
        super().__init__()
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for LoopPlugin')

        self.pre_compute.append(self.merge_deps)

    def merge_deps(self, chunk_i, **kwargs):
        return {k: strax.merge_arrs([kwargs[d] for d in deps])
                for k, deps in self.dependencies_by_kind().items()}


@export
class LoopPlugin(MergePlugin):
    """Plugin that disguises multi-kind data-iteration by an event loop
    """
    def compute(self, **kwargs):
        # If not otherwise specified, data kind to loop over
        # is that of the first dependency (e.g. events)
        # Can't be in __init__: deps not initialized then
        if hasattr(self, 'loop_over'):
            loop_over = self.loop_over
        else:
            loop_over = self.deps[self.depends_on[0]].data_kind

        # Group into lists of things (e.g. peaks)
        # contained in the base things (e.g. events)
        base = kwargs[loop_over]
        assert np.all(base[1:]['time'] >= strax.endtime(base[:-1])), \
            f'{base}s overlap'

        for k, things in kwargs.items():
            assert np.diff(things['time']).min() > 0, f'{k} not sorted'
            if k != loop_over:
                r = strax.split_by_containment(things, base)
                if len(r) != len(base):
                    raise RuntimeError(f"Split {k} into {len(r)}, "
                                       f"should be {len(base)}!")
                kwargs[k] = r

        results = np.zeros(len(base), dtype=self.dtype)
        for i in range(len(base)):
            r = self.compute_loop(base[i],
                                  **{k: kwargs[k][i]
                                     for k in self.dependencies_by_kind()
                                     if k != loop_over})

            # Convert from dict to array row:
            for k, v in r.items():
                results[i][k] = v

        return results

    def compute_loop(self, *args, **kwargs):
        raise NotImplementedError


@export
class MergeOnlyPlugin(MergePlugin):
    """Plugin that merges data from its dependencies
    """
    save_when = SaveWhen.EXPLICIT

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergePlugins can only merge data of the same "
                             "kind, but got multiple kinds: "
                             + str(deps_by_kind))

        return sum([strax.unpack_dtype(self.deps[d].dtype)
                    for d in self.depends_on], [])

    def compute(self, **kwargs):
        return strax.merge_arrs(list(kwargs.values()))


@export
class PlaceholderPlugin(Plugin):
    """Plugin that throws NotImplementedError when asked to compute anything"""
    depends_on = tuple()

    def compute(self):
        raise NotImplementedError("No plugin registered that "
                                  f"provides {self.provides}")


@export
class RecordsPlaceholder(PlaceholderPlugin):
    """Placeholder plugin for something (e.g. a DAQ or simulator) that
    provides strax records.
    """
    provides = 'records'
    data_kind = 'records'
    dtype = strax.record_dtype()
