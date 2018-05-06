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

    rechunk_on_save = True    # Saver is allowed to rechunk
    rechunk_input = False

    save_when = SaveWhen.ALWAYS
    parallel = False    # If True, compute() work is submitted to pool
    save_meta_only = False

    # These are set on plugin initialization, which is done in the core
    run_id: str
    config: typing.Dict
    deps: typing.List       # Dictionary of dependency plugin instances
    takes_config = dict()       # Config options
    compute_takes_chunk_i = False    # Autoinferred, no need to set yourself

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on not provided for '
                             f'{self.__class__.__name__}')
        self.post_compute = []
        self.on_close = []
        if self.rechunk_input and self.parallel:
            raise RuntimeError("Plugins that rechunk their input "
                               "cannot be parallelized")

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

    def dependencies_by_kind(self, require_time=None):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True, one dependency of each kind
        must provide time information. It will be put first in the list.

        If require_time is omitted, we will require time only if there is
        more than one data kind in the dependencies.
        """
        if require_time is None:
            require_time = \
                len(self.dependencies_by_kind(require_time=False)) > 1

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
                    raise ValueError(f"For {self.__class__.__name__}, no "
                                     f"dependency of data kind {k} "
                                     "has time information!")

        return deps_by_kind

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results
        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to.
            If None, will compute inside the plugin's thread.
        """
        deps_by_kind = self.dependencies_by_kind()

        if len(deps_by_kind) > 1:
            # Sync the iterators that provide time info for each data kind
            # (first in deps_by_kind lists) by endtime
            iters.update(strax.sync_iters(
                partial(strax.same_stop, func=strax.endtime),
                {d[0]: iters[d[0]]
                 for d in deps_by_kind.values()}))

        # Convert to iterators over merged data for each kind
        new_iters = dict()
        for kind, deps in deps_by_kind.items():
            if len(deps) > 1:
                synced_iters = strax.sync_iters(
                    strax.same_length,
                    {d: iters[d] for d in deps})
                new_iters[kind] = strax.merge_iters(synced_iters.values())
            else:
                new_iters[kind] = iters[deps[0]]
        iters = new_iters

        if self.rechunk_input:
            iters = self.rechunk_input(iters)

        pending = []
        for chunk_i in itertools.count():
            try:
                if not self.check_next_ready_or_done(chunk_i):
                    # TODO: avoid duplication
                    # but also remain picklable...
                    self.close(wait_for=tuple(pending))
                    return
                compute_kwargs = {k: next(iters[k])
                                  for k in deps_by_kind}
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

        if self.compute_takes_chunk_i:
            result = self.compute(**kwargs, chunk_i=chunk_i)
        else:
            result = self.compute(**kwargs)

        if isinstance(result, dict):
            if not len(result):
                # TODO: alt way of getting length?
                raise RuntimeError("if returning dict, must have a key")
            some_key = list(result.keys())[0]
            n = len(result[some_key])
            r = np.zeros(n, dtype=self.dtype)
            for k, v in result.items():
                r[k] = v
            result = r

        for p in self.post_compute:
            r = p(result, chunk_i=chunk_i)
            if r is not None:
                result = r

        return result

    def check_next_ready_or_done(self, chunk_i):
        return True

    def close(self, wait_for=tuple(), timeout=120):
        done, not_done = wait(wait_for, timeout=timeout)
        if len(not_done):
            raise RuntimeError(
                f"{len(not_done)} futures of {self.__class__.__name__}"
                "did not complete in time!")
        for x in self.on_close:
            x()

    def compute(self, **kwargs):
        raise NotImplementedError

##
# Special plugins
##


@export
class LoopPlugin(Plugin):
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
        if len(base) > 1:
            assert np.all(base[1:]['time'] >= strax.endtime(base[:-1])), \
                f'{base}s overlap'

        for k, things in kwargs.items():
            if len(things) > 1:
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
class MergeOnlyPlugin(Plugin):
    """Plugin that merges data from its dependencies
    """
    save_when = SaveWhen.EXPLICIT

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergeOnlyPlugins can only merge data "
                             "of the same kind, but got multiple kinds: "
                             + str(deps_by_kind))

        return strax.merged_dtype([self.deps[d].dtype
                                   for d in self.depends_on])

    def compute(self, **kwargs):
        return kwargs[list(kwargs.keys())[0]]
