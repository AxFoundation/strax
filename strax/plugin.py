"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from enum import IntEnum
import itertools
import logging
from functools import partial
import typing
import time
import inspect

from frozendict import frozendict
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
class InputTimeoutExceeded(Exception):
    pass


@export
class PluginGaveWrongOutput(Exception):
    pass


class PendingFutures:
    def __init__(self):
        self._pending = []

    def add(self, future):
        self._pending.append(future)

    def update(self):
        self._pending = [f for f in self._pending
                         if not f.done()]

    def remaining_futures(self):
        self.update()
        return self._pending


@export
class Plugin:
    """Plugin containing strax computation

    You should NOT instantiate plugins directly.
    Do NOT add unpickleable things (e.g. loggers) as attributes.
    """
    __version__ = '0.0.0'

    # For multi-output plugins these should be (frozen)dicts
    data_kind: typing.Union[str, frozendict, dict]
    dtype: typing.Union[tuple, np.dtype, frozendict, dict]

    depends_on: tuple
    provides: tuple

    compressor = 'blosc'

    rechunk_on_save = True    # Saver is allowed to rechunk

    # For a source with online input (e.g. DAQ readers), crash if no new input
    # has appeared for this many seconds
    # This should be smaller than the mailbox timeout (which is intended as
    # a deep fallback)
    input_timeout = 80

    save_when = SaveWhen.ALWAYS

    # Instructions how to parallelize
    #   False: never parallellize;
    #   'process': use processpool;
    #   'thread' (or just True): use threadpool.
    parallel = False              # For the computation itself

    # Maximum number of output messages
    max_messages = None   # use default

    # Do not specify attributes below

    # Set using the takes_config decorator
    takes_config = frozendict()

    # These are set on plugin initialization, which is done in the core
    run_id: str
    run_i: int
    config: typing.Dict
    deps: typing.Dict       # Dictionary of dependency plugin instances
    
    compute_takes_chunk_i = False    # Autoinferred, no need to set yourself

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on not provided for '
                             f'{self.__class__.__name__}')

        self.depends_on = strax.to_str_tuple(self.depends_on)

        # Store compute parameter names, see if we take chunk_i too
        compute_pars = list(
            inspect.signature(self.compute).parameters.keys())
        if 'chunk_i' in compute_pars:
            self.compute_takes_chunk_i = True
            del compute_pars[compute_pars.index('chunk_i')]

        self.compute_pars = compute_pars

    @property
    def multi_output(self):
        return len(self.provides) > 1

    def setup(self):
        """Hook if plugin wants to do something on initialization
        """
        pass

    def infer_dtype(self):
        """Return dtype of computed data;
        used only if no dtype attribute defined"""
        # Don't raise NotImplementedError, IDE will complain you're not
        # implementing all abstract methods...
        raise RuntimeError("No infer dtype method defined")

    def version(self, run_id=None):
        """Return version number applicable to the run_id.
        Most plugins just have a single version (in .__version__)
        but some may be at different versions for different runs
        (e.g. time-dependent corrections).
        """
        return self.__version__

    def dtype_for(self, data_type):
        if self.multi_output:
            return self.dtype[data_type]
        return self.dtype

    def empty_result(self):
        if self.multi_output:
            return {d: np.empty(0, self.dtype_for(d))
                    for d in self.provides}
        return np.empty(0, self.dtype)

    def data_kind_for(self, data_type):
        if self.multi_output:
            return self.data_kind[data_type]
        return self.data_kind

    def metadata(self, run_id, data_type):
        """Metadata to save along with produced data"""
        if not data_type in self.provides:
            raise RuntimeError(f"{data_type} not in {self.provides}?")
        return dict(
            run_id=run_id,
            data_type=data_type,
            data_kind=self.data_kind_for(data_type),
            dtype=self.dtype_for(data_type),
            lineage_hash=strax.DataKey(
                run_id, data_type, self.lineage).lineage_hash,
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
        return strax.group_by_kind(
            self.depends_on,
            plugins=self.deps,
            require_time=require_time)

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading.
        Returns True by default; override if you make an online input plugin.
        """
        return True

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written.
        Only called for online input plugins.
        """
        # Don't raise NotImplementedError, IDE complains
        raise RuntimeError("source_finished called on a regular plugin")

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None,
            will compute inside the plugin's thread.
        """
        deps_by_kind = self.dependencies_by_kind()

        # Merge iterators of data that has the same kind
        kind_iters = dict()
        for kind, deps in deps_by_kind.items():
            kind_iters[kind] = strax.merge_iters(
                strax.sync_iters(
                    strax.same_length,
                    {d: iters[d] for d in deps}))

        if len(deps_by_kind) > 1:
            # Sync iterators of different kinds by time
            kind_iters = strax.sync_iters(
                partial(strax.same_stop, func=strax.endtime),
                kind_iters)

        iters = kind_iters
        pending = PendingFutures()
        yield from self._inner_iter(iters, pending, executor)
        self.cleanup(wait_for=pending.remaining_futures())

    def _inner_iter(self, iters, pending: PendingFutures, executor):
        last_input_received = time.time()

        for chunk_i in itertools.count():

            # Online input support
            while not self.is_ready(chunk_i):
                if self.source_finished():
                    print("Source finished!")
                    # Source is finished, there is no next chunk: break out
                    return

                if time.time() > last_input_received + self.input_timeout:
                    raise InputTimeoutExceeded(
                        f"{self.__class__.__name__}:{id(self)} waited for "
                        f"more  than {self.input_timeout} sec for arrival of "
                        f"input chunk {chunk_i}, and has given up.")

                print(f"{self.__class__.__name__}:{id(self)} "
                      f"waiting for chunk {chunk_i}")
                time.sleep(2)
            last_input_received = time.time()

            # Actually fetch the input from the iterators
            try:
                compute_kwargs = {k: next(iters[k])
                                  for k in iters}
            except StopIteration:
                return

            if self.parallel and executor is not None:
                new_f = executor.submit(self.do_compute,
                                        chunk_i=chunk_i,
                                        **compute_kwargs)
                pending.add(new_f)
                pending.update()
                yield new_f
            else:
                yield self.do_compute(chunk_i=chunk_i, **compute_kwargs)

    def cleanup(self, wait_for):
        pass

    def _check_dtype(self, x, d=None):
        if d is None:
            assert not self.multi_output
            d = self.provides[0]
        expect = self.dtype_for(d)
        pname = self.__class__.__name__
        if not isinstance(x, np.ndarray):
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver "
                f"data type {d} as promised.\n"
                f"Delivered a {type(x)}")
        if not isinstance(expect, np.dtype):
            raise ValueError(f"Plugin {pname} expects {expect} as dtype??")
        if x.dtype != expect:
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver "
                f"data type {d} as promised.\n"
                f"Promised: {self.dtype_for(d)}\n"
                f"Delivered: {x.dtype}.")

    def do_compute(self, chunk_i=None, **kwargs):
        """Wrapper for the user-defined compute method

        This is the 'job' that gets executed in different processes/threads
        during multiprocessing
        """
        if self.compute_takes_chunk_i:
            result = self.compute(chunk_i=chunk_i, **kwargs)
        else:
            result = self.compute(**kwargs)
        return self._fix_output(result)

    def _fix_output(self, result):
        if self.multi_output:
            if not isinstance(result, dict):
                raise ValueError(
                    f"{self.__class__.__name__} is multi-output and should "
                    "provide a dict output {dtypename: array}")
            r2 = dict()
            for d in self.provides:
                if d not in result:
                    raise ValueError(f"Data type {d} missing from output of "
                                     f"{self.__class__.__name__}!")
                r2[d] = strax.dict_to_rec(result[d], self.dtype_for(d))
                self._check_dtype(r2[d], d)
            return r2

        result = strax.dict_to_rec(result, dtype=self.dtype)
        self._check_dtype(result)
        return result

    def compute(self, **kwargs):
        raise NotImplementedError


##
# Special plugins
##

@export
class OverlapWindowPlugin(Plugin):
    """Plugin whose computation depends on having its inputs extend
    a certain window on both sides.

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
        self.last_threshold = -float('inf')
        # This guy can have a logger, it's not parallelized anyway
        self.log = logging.getLogger(self.__class__.__name__)

    def get_window_size(self):
        """Return the required window size in nanoseconds"""
        raise NotImplementedError

    def iter(self, iters, executor=None):
        yield from super().iter(iters, executor=executor)

        # Yield results initially suppressed in fear of a next chunk
        if self.cached_results is not None and len(self.cached_results):
            self.log.debug(f"Last chunk! Sending out cached result "
                           f"{self.cached_results}")
            yield self.cached_results
        else:
            self.log.debug("Last chunk! No cached results to send.")

    def do_compute(self, chunk_i=None, **kwargs):
        if not len(kwargs):
            raise RuntimeError("OverlapWindowPlugin must have a dependency")

        # Determine (a lower bound on) the data's last endtime
        ends = [strax.endtime(x[-1])
                for x in kwargs.values()
                if len(x)]
        if not len(ends):
            # Input is completely empty, we cannot estimate the data's end.
            # Do not discard or send anything until a chunk with data arrives.
            # (or the last chunk, see iter)
            invalid_beyond = cache_inputs_beyond = self.last_threshold
        else:
            # Take slightly larger windows for safety: it is very easy for me
            # (or the user) to have made an off-by-one error
            # TODO: why do tests not fail is I set cache_inputs_beyond to
            # end - window size - 2 ?
            # (they do fail if I set to end - 0.5 * window size - 2)
            end = max(ends)
            invalid_beyond = end - self.get_window_size() - 1
            cache_inputs_beyond = end - 2 * self.get_window_size() - 1

        # Update input cache
        for k, v in kwargs.items():
            if len(self.cached_input):
                kwargs[k] = v = np.concatenate([self.cached_input[k], v])
            self.cached_input[k] = v[strax.endtime(v) > cache_inputs_beyond]

        if not len(ends):
            # Output cache and last_threshold both don't change
            # so might as well return.
            return self.empty_result()

        result = super().do_compute(chunk_i=chunk_i, **kwargs)

        endtimes = strax.endtime(kwargs[self.data_kind]
                                 if self.data_kind in kwargs
                                 else result)
        assert len(endtimes) == len(result)

        is_valid = endtimes < invalid_beyond
        not_sent_yet = endtimes >= self.last_threshold

        # Cache all results we have not sent, nor are sending now
        self.cached_results = result[not_sent_yet & (~is_valid)]

        # Send out only valid results we haven't sent yet
        result = result[is_valid & not_sent_yet]

        self.last_threshold = invalid_beyond
        return result


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
            # Check for sorting
            difs = np.diff(things['time'])
            if difs.min(initial=0) < 0:
                i_bad = np.argmin(difs)
                examples = things[i_bad-1:i_bad+3]
                t0 = examples['time'].min()
                raise ValueError(
                    f'Expected {k} to be sorted, but found ' +
                    str([(x['time'] - t0, strax.endtime(x) - t0)
                         for x in examples]))

            if k != loop_over:
                r = strax.split_by_containment(things, base)
                if len(r) != len(base):
                    raise RuntimeError(f"Split {k} into {len(r)}, "
                                       f"should be {len(base)}!")
                kwargs[k] = r

        results = np.zeros(len(base), dtype=self.dtype)
        deps_by_kind = self.dependencies_by_kind()

        for i in range(len(base)):
            r = self.compute_loop(base[i],
                                  **{k: kwargs[k][i]
                                     for k in deps_by_kind
                                     if k != loop_over})

            # Convert from dict to array row:
            for k, v in r.items():
                results[i][k] = v

        return results

    def compute_loop(self, *args, **kwargs):
        raise NotImplementedError


##
# "Plugins" for internal use
# These do not actually do computations, but do other tasks
# for which posing as a plugin is helpful.
# Do not subclass unless you know what you are doing..
##

@export
class MergeOnlyPlugin(Plugin):
    """Plugin that merges data from its dependencies
    """
    save_when = SaveWhen.NEVER

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergeOnlyPlugins can only merge data "
                             "of the same kind, but got multiple kinds: "
                             + str(deps_by_kind))

        return strax.merged_dtype([self.deps[d].dtype_for(d)
                                   for d in self.depends_on])

    def compute(self, **kwargs):
        return kwargs[list(kwargs.keys())[0]]


@export
class ParallelSourcePlugin(Plugin):
    """An plugin that inlines the computation of other plugins and saving
    of their results.

    This evades data transfer (pickling and/or memory copy) penalties.
    """
    parallel = 'process'

    @classmethod
    def inline_plugins(cls, components, start_from, log):
        plugins = components.plugins.copy()

        sub_plugins = {start_from: plugins[start_from]}
        del plugins[start_from]

        # Gather all plugins that do not rechunk and which branch out as a
        # simple tree from the input plugin.
        # We'll run these all together in one process.
        while True:
            # Scan for plugins we can inline
            for p in plugins.values():
                if (p.parallel
                        and all([d in sub_plugins for d in p.depends_on])):
                    for d in p.provides:
                        sub_plugins[d] = p
                        if d in plugins:
                            del plugins[d]
                    # Rescan
                    break
            else:
                # No more plugins we can inline
                break

        if len(set(list(sub_plugins.values()))) == 1:
            # Just one plugin to inline: no use
            log.debug("Just one plugin to inline: skipping")
            return components

        # Which data types should we output? Three cases follow.
        outputs_to_send = set()

        # Case 1. Requested as a final target
        for p in sub_plugins.values():
            outputs_to_send.update(set(components.targets)
                                   .intersection(set(p.provides)))
        # Case 2. Requested by a plugin we did not inline
        for d, p in plugins.items():
            outputs_to_send.update(set(p.depends_on))
        outputs_to_send &= sub_plugins.keys()

        # Inline savers that do not require rechunking
        savers = components.savers
        sub_savers = dict()
        for d, p in sub_plugins.items():

            if d not in savers:
                continue
            if p.rechunk_on_save:
                # Case 3. has a saver we can't inline (this is checked later)
                outputs_to_send.add(d)
                continue

            remaining_savers = []
            for s_i, s in enumerate(savers[d]):
                if not s.allow_fork:
                    # Case 3 again, cannot inline saver
                    outputs_to_send.add(d)
                    remaining_savers.append(s)
                    continue
                if d not in sub_savers:
                    sub_savers[d] = []
                s.is_forked = True
                sub_savers[d].append(s)
            savers[d] = remaining_savers

            if not len(savers[d]):
                del savers[d]

        p = cls(depends_on=sub_plugins[start_from].depends_on)
        p.sub_plugins = sub_plugins
        p.provides = tuple(outputs_to_send)
        p.sub_savers = sub_savers
        p.start_from = start_from
        if p.multi_output:
            p.dtype = {d: p.sub_plugins[d].dtype_for(d)
                       for d in outputs_to_send}
        else:
            to_send = list(outputs_to_send)[0]
            p.dtype = p.sub_plugins[to_send].dtype_for(to_send)
        for d in p.provides:
            plugins[d] = p
        p.deps = {d: plugins[d] for d in p.depends_on}

        log.debug(f"Inlined plugins: {p.sub_plugins}."
                  f"Inlined savers: {p.sub_savers}")

        return strax.ProcessorComponents(
            plugins, components.loaders, savers, components.targets)

    def __init__(self, depends_on):
        self.depends_on = depends_on
        super().__init__()

    def source_finished(self):
        return self.sub_plugins[self.start_from].source_finished()

    def is_ready(self, chunk_i):
        return self.sub_plugins[self.start_from].is_ready(chunk_i)

    def do_compute(self, chunk_i=None, **kwargs):
        results = kwargs

        # Run the different plugin computations
        while True:
            for output_name, p in self.sub_plugins.items():
                if output_name in results:
                    continue
                # Sorting deps since otherwise input field order depends on
                # order in which computation happened, which might be bad?
                deps = sorted(p.depends_on)
                if any([d not in results for d in deps]):
                    continue
                compute_kwargs = dict(chunk_i=chunk_i)

                for kind, d_of_kind in p.dependencies_by_kind().items():
                    compute_kwargs[kind] = strax.merge_arrs(
                        [results[d] for d in d_of_kind])

                # Store compute result(s)
                r = p.do_compute(**compute_kwargs)
                if p.multi_output:
                    for d in r:
                        results[d] = r[d]
                else:
                    results[output_name] = r

                # Rescan plugins to see if we can compute anything more
                break

            else:
                # Nothing further to compute
                break
        for d in self.provides:
            assert d in results, f"Output {d} missing!"

        # Save anything we can through the inlined savers
        for d, savers in self.sub_savers.items():
            for s in savers:
                s.save(data=results[d], chunk_i=chunk_i)

        # Remove results we do not need to send
        for d in list(results.keys()):
            if d not in self.provides:
                del results[d]

        if not self.multi_output:
            results = results[self.provides[0]]

        return self._fix_output(results)

    def cleanup(self, wait_for):
        print(f"{self.__class__.__name__} exhausted. "
              f"Waiting for {len(wait_for)} pending futures.")
        for savers in self.sub_savers.values():
            for s in savers:
                s.close(wait_for=wait_for)
