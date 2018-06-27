"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from enum import IntEnum
import itertools
from functools import partial
import typing
import time
import inspect

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
    # Child class can make this a function that takes and returns iters:
    rechunk_input = None

    save_when = SaveWhen.ALWAYS
    parallel = False    # If True, compute() work is submitted to pool
    save_meta_only = False

    # These are set on plugin initialization, which is done in the core
    run_id: str
    run_i: int
    config: typing.Dict
    deps: typing.List       # Dictionary of dependency plugin instances
    compute_takes_chunk_i = False    # Autoinferred, no need to set yourself
    takes_config = dict()       # Config options

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on not provided for '
                             f'{self.__class__.__name__}')
        if self.rechunk_input and self.parallel:
            raise RuntimeError("Plugins that rechunk their input "
                               "cannot be parallelized")

        # Store compute parameter names, see if we take chunk_i too
        compute_pars = list(
            inspect.signature(self.compute).parameters.keys())
        if 'chunk_i' in compute_pars:
            self.compute_takes_chunk_i = True
            del compute_pars[compute_pars.index('chunk_i')]
        self.compute_pars = compute_pars

    def setup(self):
        """Hook if plugin wants to do something on initialization
        """
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

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading.
        Returns True by default; override if you make an online input plugin.
        """
        return True

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written.
        Only called for online input plugins.
        """
        raise NotImplementedError

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None,
            will compute inside the plugin's thread.
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
            if not self.is_ready(chunk_i):
                if self.source_finished():
                    break
                print(f"{self.__class__.__name__} waiting for chunk {chunk_i}")
                time.sleep(2)

            try:
                compute_kwargs = {k: next(iters[k])
                                  for k in deps_by_kind}
            except StopIteration:
                return
            except Exception:
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

        self.cleanup(wait_for=pending)

    def cleanup(self, wait_for):
        pass

    def do_compute(self, chunk_i=None, **kwargs):
        if self.compute_takes_chunk_i:
            result = self.compute(chunk_i=chunk_i, **kwargs)
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

        return result

    def compute(self, **kwargs):
        raise NotImplementedError


##
# Special plugins
##

@export
class ParallelSourcePlugin(Plugin):
    """An input plugin that reads chunks in parallel in different processes.

    We will try to run dependencies in the same process, to evade
    data transfer (pickling and memory copy) penalties.

    Child must implement source_finished and is_ready.
    """
    parallel = 'process'

    sub_plugins: typing.Dict[str, Plugin]
    sub_savers: typing.Dict[str, typing.List[strax.Saver]]
    outputs_to_send: typing.Set[str]

    def __init__(self):
        super().__init__()
        # Subsidiary plugins and savers.
        # Dictionary provides -> plugin/saver
        self.sub_plugins = {}
        self.sub_savers = {}
        # List of ouputs to send
        self.outputs_to_send = set()

    def setup_mailboxes(self, components, mailboxes, executor):
        """Setup this plugin inside a ThreadedMailboxProcessor
        This will gather as much plugins/savers as possible as "subsidiaries"
        which can run in the same processes as the input.
        :return: ProcessorComponents, altered by setup process.
        """
        plugins = components.plugins
        savers = components.savers

        del plugins[self.provides]

        # Gather all plugins that do not rechunk and which branch out as a
        # simple tree from the input plugin.
        # We'll run these all together in one process.
        while True:
            for d, p in plugins.items():
                i_have = [self.provides] + list(self.sub_plugins.keys())
                if (len(p.depends_on) == 1
                        and p.depends_on[0] in i_have
                        and p.parallel):
                    self.sub_plugins[d] = p

                    del plugins[d]
                    break
            else:
                break

        # Which data types are we outputting?
        for d, p in plugins.items():
            self.outputs_to_send.update(
                set(p.depends_on) & self.sub_plugins.keys())

        # If the savers do not require rechunking, run them in this way also
        for d in list(self.sub_plugins.keys()) + [self.provides]:
            if d in savers:

                # Get the plugin... awkward...
                if d in self.sub_plugins:
                    p = self.sub_plugins[d]
                elif d in plugins:
                    p = plugins[d]
                elif d == self.provides:
                    p = self
                else:
                    raise RuntimeError

                if p.rechunk_on_save:
                    self.outputs_to_send.add(d)
                else:
                    self.sub_savers[d] = savers[d]
                    del savers[d]

        mailboxes[self.provides].add_sender(self.iter(
            iters={}, executor=executor))
        mailboxes[self.provides].add_reader(partial(self.send_outputs,
                                                    mailboxes=mailboxes))
        return components

    def send_outputs(self, source, mailboxes):
        for result in source:
            for d, x in result.items():
                mailboxes[d].send(x)

        for d in self.outputs_to_send:
            mailboxes[d].close()

    def cleanup(self, wait_for):
        print(f"{self.__class__.__name__} exhausted. "
              f"Waiting for {len(wait_for)} pending futures.")
        for savers in self.sub_savers.values():
            for s in savers:
                s.close(wait_for=wait_for)

    def do_compute(self, *args, chunk_i=None, **kwargs):
        result = super().do_compute(*args, chunk_i=chunk_i, **kwargs)
        # Fortunately everybody else is in a different process...
        self._output = {}
        self._grok(self.provides, result, chunk_i)
        for d in self.outputs_to_send:
            assert d in self._output, f"Output {d} missing!"
        return self._output

    def _grok(self, d, x, chunk_i):
        """Launch any computations depending on result d:x (data type:array)"""
        for other_d, p in self.sub_plugins.items():
            if p.depends_on[0] == d:
                self._grok(other_d, p.do_compute(x), chunk_i)

        if d in self.sub_savers:
            for s in self.sub_savers[d]:
                s.save(data=x, chunk_i=chunk_i)

        if d in self.outputs_to_send:
            self._output[d] = x


@export
class OverlapWindowPlugin(Plugin):
    """Plugin whose computation depends on having its inputs extend
    a certain window on both sides.

    Current implementation assumes:
    - All inputs are sorted by endtime. Since everything in strax is sorted by
    time, this only works for disjoint intervals such as peaks or events,
    but NOT records!
    - You must read time info for your data kind, or create a new data kind.
    """
    parallel = False

    def __init__(self):
        super().__init__()
        self.cached_input = {}
        self.cached_results = None
        self.last_threshold = -float('inf')

    def get_window_size(self):
        """Return the required window size in nanoseconds"""
        raise NotImplementedError

    def iter(self, iters, executor=None):
        yield from super().iter(iters, executor=executor)

        # Yield results initially suppressed in fear of a next chunk
        if self.cached_results is not None and len(self.cached_results):
            yield self.cached_results

    def do_compute(self, chunk_i=None, **kwargs):
        if not len(kwargs):
            raise RuntimeError("OverlapWindowPlugin must have a dependency")
        end = max([strax.endtime(x[-1])
                   for x in kwargs.values()])
        invalid_beyond = end - self.get_window_size()
        cache_inputs_beyond = end - 3 * self.get_window_size()

        for k, v in kwargs.items():
            if len(self.cached_input):
                kwargs[k] = v = np.concatenate([self.cached_input[k], v])
            self.cached_input[k] = v[strax.endtime(v) > cache_inputs_beyond]

        result = super().do_compute(chunk_i=chunk_i, **kwargs)

        endtimes = strax.endtime(kwargs[self.data_kind]
                                 if self.data_kind in kwargs
                                 else result)
        assert len(endtimes) == len(result)

        # Remove results that are invalid or already sent out last time
        is_valid = endtimes < invalid_beyond
        self.cached_results = result[~is_valid]
        result = result[is_valid
                        & (endtimes > self.last_threshold)]

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
