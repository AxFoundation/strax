
import logging
import typing as ty

import strax
from ..plugin import Plugin

export, __all__ = strax.exporter()


@export
class ProcessorComponents(ty.NamedTuple):
    """Specification to assemble a processor"""
    plugins: ty.Dict[str, strax.Plugin]
    loaders: ty.Dict[str, callable]
    savers:  ty.Dict[str, ty.List[strax.Saver]]
    targets: ty.Tuple[str]
    
@export
class BaseProcessor:
    components: ProcessorComponents

    def __init__(self,
                 components: ProcessorComponents,
                 allow_rechunk=True, allow_shm=False,
                 allow_multiprocess=False,
                 allow_lazy=True,
                 max_workers=None,
                 max_messages=4,
                 timeout=60):
        self.log = logging.getLogger(self.__class__.__name__)
        self.components = components

    def iter(self):
        raise NotImplementedError

class PluginRunner:
    deps: ty.Dict
    config: ty.Dict
    input_buffer: ty.Dict
    plugin: Plugin

    def __init__(self, plugin, deps):
        self.plugin = plugin
        self.deps = deps

    def dependencies_by_kind(self):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True, one dependency of each kind
        must provide time information. It will be put first in the list.

        If require_time is omitted, we will require time only if there is
        more than one data kind in the dependencies.
        """
        return strax.group_by_kind(
            self.depends_on,
            plugins=self.deps)

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

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        """Add a chunk of the datatype d to the input buffer.
        Return True if this succeeded, False if the source is exhausted.
        :param d: data type to fetch
        :param iters: iterators that produce data
        :param check_end_not_before: Raise a runtimeError if the source 
        is exhausted, but the input buffer ends before this time.
        """
        try:
            # print(f"Fetching {d} in {self}, hope to see {hope_to_see}")
            self.input_buffer[d] = strax.Chunk.concatenate(
                [self.input_buffer[d], next(iters[d])])
            # print(f"Fetched {d} in {self}, "
            #      f"now have {self.input_buffer[d]}")
            return True
        except StopIteration:
            # print(f"Got StopIteration while fetching for {d} in {self}")
            if (check_end_not_before is not None
                    and self.input_buffer[d].end < check_end_not_before):
                raise RuntimeError(
                    f"Tried to get data until {check_end_not_before}, but {d} "
                    f"ended prematurely at {self.input_buffer[d].end}")
            return False


    def iter(self, iters, executor=None):
        raise NotImplementedError

    def cleanup(self, wait_for):
        self.plugin.cleanup(wait_for)

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None,
            will compute inside the plugin's thread.
        """
        pending_futures = []
        last_input_received = time.time()
        self.input_buffer = {d: None
                             for d in self.depends_on}

        # Fetch chunks from all inputs. Whoever is the slowest becomes the
        # pacemaker
        pacemaker = None
        _end = float('inf')
        for d in self.depends_on:
            self._fetch_chunk(d, iters)
            if self.input_buffer[d] is None:
                raise ValueError(f'Cannot work with empty input buffer {self.input_buffer}')
            if self.input_buffer[d].end < _end:
                pacemaker = d
                _end = self.input_buffer[d].end

        # To break out of nested loops:
        class IterDone(Exception):
            pass

        try:
            for chunk_i in itertools.count():

                # Online input support
                while not self.is_ready(chunk_i):
                    if self.source_finished():
                        # Chunk_i does not exist. We are done.
                        print("Source finished!")
                        raise IterDone()

                    if time.time() > last_input_received + self.input_timeout:
                        raise InputTimeoutExceeded(
                            f"{self.__class__.__name__}:{id(self)} waited for "
                            f"more  than {self.input_timeout} sec for arrival of "
                            f"input chunk {chunk_i}, and has given up.")

                    print(f"{self.__class__.__name__} with object id: {id(self)} "
                          f"waits for chunk {chunk_i}")
                    time.sleep(2)
                last_input_received = time.time()

                if pacemaker is None:
                    inputs_merged = dict()
                else:
                    if chunk_i != 0:
                        # Fetch the pacemaker, to figure out when this chunk ends
                        # (don't do it for chunk 0, for which we already fetched)
                        if not self._fetch_chunk(pacemaker, iters):
                            # Source exhausted. Cleanup will do final checks.
                            raise IterDone()
                    this_chunk_end = self.input_buffer[pacemaker].end

                    inputs = dict()
                    # Fetch other inputs (when needed)
                    for d in self.depends_on:
                        if d != pacemaker:
                            while (self.input_buffer[d] is None
                                   or self.input_buffer[d].end < this_chunk_end):
                                self._fetch_chunk(
                                    d, iters,
                                    check_end_not_before=this_chunk_end)
                        inputs[d], self.input_buffer[d] = \
                            self.input_buffer[d].split(
                                t=this_chunk_end,
                                allow_early_split=True)
                    # If any of the inputs were trimmed due to early splits,
                    # trim the others too.
                    # In very hairy cases this can take multiple passes.
                    # TODO: can we optimize this, or code it more elegantly?
                    max_passes_left = 10
                    while max_passes_left > 0:
                        this_chunk_end = min([x.end for x in inputs.values()]
                                             + [this_chunk_end])
                        if len(set([x.end for x in inputs.values()])) <= 1:
                            break
                        for d in self.depends_on:
                            inputs[d], back_to_buffer = \
                                inputs[d].split(
                                    t=this_chunk_end,
                                    allow_early_split=True)
                            self.input_buffer[d] = strax.Chunk.concatenate(
                                [back_to_buffer, self.input_buffer[d]])
                        max_passes_left -= 1
                    else:
                        raise RuntimeError(
                            f"{self} was unable to get time-consistent "
                            f"inputs after ten passess. Inputs: \n{inputs}\n"
                            f"Input buffer:\n{self.input_buffer}")

                    # Merge inputs of the same kind
                    inputs_merged = {
                        kind: strax.Chunk.merge([inputs[d] for d in deps_of_kind])
                        for kind, deps_of_kind in self.dependencies_by_kind().items()}

                # Submit the computation
                # print(f"{self} calling with {inputs_merged}")
                if self.parallel and executor is not None:
                    new_future = executor.submit(
                        self.do_compute,
                        chunk_i=chunk_i,
                        **inputs_merged)
                    pending_futures.append(new_future)
                    pending_futures = [f for f in pending_futures if not f.done()]
                    yield new_future
                else:
                    yield self.do_compute(chunk_i=chunk_i, **inputs_merged)

        except IterDone:
            # Check all sources are exhausted.
            # This is more than a check though -- it ensure the content of
            # all sources are requested all the way (including the final
            # Stopiteration), as required by lazy-mode processing requires
            for d in iters.keys():
                if self._fetch_chunk(d, iters):
                    raise RuntimeError(
                        f"Plugin {d} terminated without fetching last {d}!")

            # This can happen especially in time range selections
            if int(self.save_when) != strax.SaveWhen.NEVER:
                for d, buffer in self.input_buffer.items():
                    # Check the input buffer is empty
                    if buffer is not None and len(buffer):
                        raise RuntimeError(
                            f"Plugin {d} terminated with leftover {d}: {buffer}")

        finally:
            self.cleanup(wait_for=pending_futures)

