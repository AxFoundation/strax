"""Plugin system for strax.

A 'plugin' is something that outputs an array and gets arrays from one or more other plugins.

"""

from enum import IntEnum
import inspect
import itertools
import logging
import time
import typing
from warnings import warn
from immutabledict import immutabledict
import numpy as np
from copy import copy, deepcopy
import strax

export, __all__ = strax.exporter()


LOGGERS = {}


@export
class SaveWhen(IntEnum):
    """Plugin's preference for having it's data saved."""

    NEVER = 0  # Throw an error if the user lists it
    EXPLICIT = 1  # Save ONLY if the user lists it explicitly
    TARGET = 2  # Save if the user asks for it as a final target
    ALWAYS = 3  # Save even if the user does not list it


@export
class InputTimeoutExceeded(Exception):
    pass


@export
class PluginGaveWrongOutput(Exception):
    pass


@export
class Plugin:
    """Plugin containing strax computation.

    You should NOT instantiate plugins directly. Do NOT add unpickleable things (e.g. loggers) as
    attributes.

    """

    __version__: typing.Optional[str] = "0.0.0"

    # For multi-output plugins these should be (immutable)dicts
    data_kind: typing.Union[str, immutabledict, dict]
    dtype: typing.Union[tuple, np.dtype, immutabledict, dict]

    depends_on: tuple
    provides: tuple
    input_buffer: typing.Dict[str, strax.Chunk]

    # Needed for plugins which are inherited from an already existing plugins,
    # indicates such an inheritance.
    child_plugin = False

    compressor = "blosc"

    rechunk_on_save = True  # Saver is allowed to rechunk
    # How large (uncompressed) should re-chunked chunks be?
    # Meaningless if rechunk_on_save is False
    chunk_target_size_mb = strax.default_chunk_size_mb

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
    parallel: typing.Union[str, bool] = False  # For the computation itself

    # Maximum number of output messages
    max_messages = None  # use default

    # Do not specify attributes below

    # Set using the takes_config decorator
    takes_config = immutabledict()

    # These are set on plugin initialization, which is done in the core
    run_id: str
    run_i: int
    config: typing.Dict
    deps: typing.Dict  # Dictionary of dependency plugin instances

    compute_takes_chunk_i = False  # Autoinferred, no need to set yourself
    compute_takes_start_end = False

    def __init__(self):
        if not hasattr(self, "depends_on"):
            raise ValueError(f"depends_on not provided for {self.__class__.__name__}")

        self.depends_on = strax.to_str_tuple(self.depends_on)

        # Store compute parameter names, see if we take chunk_i too
        compute_pars = list(inspect.signature(self.compute).parameters.keys())
        if "chunk_i" in compute_pars:
            self.compute_takes_chunk_i = True
            del compute_pars[compute_pars.index("chunk_i")]
        if "start" in compute_pars:
            if "end" not in compute_pars:
                raise ValueError(f"Compute of {self} takes start, so it should also take end.")
            self.compute_takes_start_end = True
            del compute_pars[compute_pars.index("start")]
            del compute_pars[compute_pars.index("end")]

        if not isinstance(self.save_when, (IntEnum, immutabledict, int)):
            raise ValueError(
                "save_when must be either a SaveWhen object or an immutabledict "
                "representing the different data_types provided."
            )

        if hasattr(self, "provides") and not isinstance(self.save_when, immutabledict):
            # The ParallelSource plugin does not provide anything as it
            # inlines only already existing components, therefore we also do
            # not have to updated save_when
            self.save_when = immutabledict.fromkeys(self.provides, self.save_when)

        if getattr(self, "provides", None):
            self.provides = strax.to_str_tuple(self.provides)
        self.compute_pars = compute_pars
        self.input_buffer = dict()

    def __copy__(self, _deep_copy=False):
        """Copy main attributes that are set after __init__ by the context.

        Note:
            self.deps is NOT copied for it is recursive and therefor slow.
            Instead, this is better handled within the context after all
            plugins are copied.

        """
        plugin_copy = self.__class__()
        plugin_copy.__init__()
        # As explained in PR #485 only copy attributes whereof we know
        # don't depend on the run_id (for use_per_run_defaults == False).
        # Otherwise we might copy run-dependent things like to_pe.
        for attribute in [
            "dtype",
            "lineage",
            "takes_config",
            "__version__",
            "config",
            "data_kind",
            "deps",
        ]:
            source_value = getattr(self, attribute)
            if _deep_copy:
                plugin_copy.__setattr__(attribute, deepcopy(source_value))
            else:
                plugin_copy.__setattr__(attribute, copy(source_value))
        return plugin_copy

    def __deepcopy__(self):
        return self.__copy__(_deep_copy=True)

    def __getattr__(self, name):
        """Allow access to config parameters as attributes this allows backwards compatibility in
        cases where a descriptor style config depends on a non descriptor style config."""
        if name == "config":
            raise AttributeError("Plugin not configured yet.")
        if hasattr(self, "config") and name in self.config:
            message = """
            Looks like you are mixing config paradigms,
            this is not recommended.
            """
            warn(message, UserWarning)
            if isinstance(self.takes_config[name], strax.Config):
                return self.takes_config[name].__get__(self)
            return self.config[name]

        raise AttributeError(f"{self.__class__.__name__} instance has no attribute {name}")

    def fix_dtype(self):
        try:
            # Infer dtype should always precede self.dtype (e.g. due to
            # copying)
            self.dtype = self.infer_dtype()
        except RuntimeError:
            if not hasattr(self, "dtype"):
                raise NotImplementedError(f"No dtype or infer_dtype specified")

        if self.multi_output:
            # Convert to a dict of numpy dtypes
            if not hasattr(self, "data_kind") or not isinstance(
                self.data_kind, (dict, immutabledict)
            ):
                raise ValueError(
                    f"{self.__class__.__name__} has multiple outputs and "
                    "must declare its data kind as a dict: "
                    "{dtypename: data kind}."
                )
            if not isinstance(self.dtype, dict):
                raise ValueError(
                    f"{self.__class__.__name__} has multiple outputs, so its "
                    "dtype must be specified as a dict: {output: dtype}."
                )
            self.dtype = {k: strax.to_numpy_dtype(dt) for k, dt in self.dtype.items()}
        else:
            # Convert to a numpy dtype
            self.dtype = strax.to_numpy_dtype(self.dtype)

        # Check required time information is present
        for d in self.provides:
            fieldnames = self.dtype_for(d).names
            ok = "time" in fieldnames and (
                ("dt" in fieldnames and "length" in fieldnames) or "endtime" in fieldnames
            )
            if not ok:
                raise ValueError(f"Missing time and endtime information for {d}")

    @property
    def multi_output(self):
        return len(self.provides) > 1

    @property
    def log(self):
        _id = id(self)
        if _id not in LOGGERS:
            LOGGERS[_id] = logging.getLogger(self.__class__.__name__)
        return LOGGERS[_id]

    def setup(self):
        """Hook if plugin wants to do something on initialization."""
        pass

    def infer_dtype(self):
        """Return dtype of computed data; used only if no dtype attribute defined."""
        # Don't raise NotImplementedError, IDE will complain you're not
        # implementing all abstract methods...
        raise RuntimeError("No infer dtype method defined")

    @property
    def _auto_version(self):
        """Generate some auto-incremented version for the context hashing system, see
        github.com/AxFoundation/strax/issues/217.

        Activate with setting __version__ to None

        """
        attributes = [attr for attr in self.__dir__() if not attr.startswith("__")]

        def _return_hashable(attr):
            if attr in ["takes_config", "_auto_version"]:
                # handled by context (or not worth tracking)
                return
            obj = getattr(self, attr)
            try:
                return strax.deterministic_hash(inspect.getsource(obj))
            except TypeError:
                pass
            try:
                return strax.deterministic_hash(obj)
            except TypeError:
                return str(obj)

        res = {attr: _return_hashable(attr) for attr in attributes}
        return "auto_" + strax.deterministic_hash(res)

    def version(self, run_id=None):
        """Return version number applicable to the run_id.

        Most plugins just have a single version (in .__version__) but some may be at different
        versions for different runs (e.g. time-dependent corrections).

        """
        if self.__version__ is None:
            return self._auto_version
        return self.__version__

    def __repr__(self):
        return self.__class__.__name__

    def dtype_for(self, data_type):
        """Provide the dtype of one of the provide arguments of the plugin.

        NB: does not simply provide the dtype of any datatype but must
        be one of the provide arguments known to the plugin.

        """
        if self.multi_output:
            if data_type in self.dtype:
                return self.dtype[data_type]
            else:
                raise ValueError(
                    "dtype_for provides the dtype of one of the "
                    "provide datatypes specified in this plugin "
                    f"{data_type} is not provided by this plugin"
                )
        return self.dtype

    def can_rechunk(self, data_type):
        if isinstance(self.rechunk_on_save, bool):
            return self.rechunk_on_save
        if isinstance(self.rechunk_on_save, (dict, immutabledict)):
            return self.rechunk_on_save[data_type]
        raise ValueError("rechunk_on_save must be a bool or an immutabledict")

    def empty_result(self):
        if self.multi_output:
            return {d: np.empty(0, self.dtype_for(d)) for d in self.provides}
        return np.empty(0, self.dtype)

    def data_kind_for(self, data_type):
        if self.multi_output:
            return self.data_kind[data_type]
        return self.data_kind

    def metadata(self, run_id, data_type):
        """Metadata to save along with produced data."""
        if data_type not in self.provides:
            raise RuntimeError(f"{data_type} not in {self.provides}?")
        return dict(
            run_id=run_id,
            data_type=data_type,
            data_kind=self.data_kind_for(data_type),
            dtype=self.dtype_for(data_type),
            lineage_hash=strax.DataKey(run_id, data_type, self.lineage).lineage_hash,
            compressor=self.compressor,
            lineage=self.lineage,
            chunk_target_size_mb=self.chunk_target_size_mb,
        )

    def dependencies_by_kind(self):
        """Return dependencies grouped by data kind i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}

        :param require_time: If True, one dependency of each kind must provide time information. It
            will be put first in the list. If require_time is omitted, we will require time only if
            there is more than one data kind in the dependencies.

        """
        return strax.group_by_kind(self.depends_on, plugins=self.deps)

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
        """Add a chunk of the datatype d to the input buffer. Return True if this succeeded, False
        if the source is exhausted.

        :param d: data type to fetch
        :param iters: iterators that produce data
        :param check_end_not_before: Raise a runtimeError if the source is exhausted, but the input
            buffer ends before this time.

        """
        try:
            # print(f"Fetching {d} in {self}, hope to see {hope_to_see}")
            self.input_buffer[d] = strax.Chunk.concatenate([self.input_buffer[d], next(iters[d])])
            # print(f"Fetched {d} in {self}, "
            #      f"now have {self.input_buffer[d]}")
            return True
        except StopIteration:
            # print(f"Got StopIteration while fetching for {d} in {self}")
            if check_end_not_before is not None and self.input_buffer[d].end < check_end_not_before:
                raise RuntimeError(
                    f"Tried to get data until {check_end_not_before}, but {d} "
                    f"ended prematurely at {self.input_buffer[d].end}"
                )
            return False

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results.

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None, will compute inside the
            plugin's thread.

        """
        pending_futures = []
        last_input_received = time.time()
        self.input_buffer = {d: None for d in self.depends_on}

        # Fetch chunks from all inputs. Whoever is the slowest becomes the
        # pacemaker
        pacemaker = None
        _end = float("inf")
        for d in self.depends_on:
            self._fetch_chunk(d, iters)
            if self.input_buffer[d] is None:
                raise ValueError(f"Cannot work with empty input buffer {self.input_buffer}")
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
                            f"more than {self.input_timeout} sec for arrival of "
                            f"input chunk {chunk_i}, and has given up."
                        )

                    print(
                        f"{self.__class__.__name__} with object id: {id(self)} "
                        f"waits for chunk {chunk_i}"
                    )
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
                            while (
                                self.input_buffer[d] is None
                                or self.input_buffer[d].end < this_chunk_end
                            ):
                                self._fetch_chunk(d, iters, check_end_not_before=this_chunk_end)
                        inputs[d], self.input_buffer[d] = self.input_buffer[d].split(
                            t=this_chunk_end, allow_early_split=True
                        )
                    # If any of the inputs were trimmed due to early splits,
                    # trim the others too.
                    # In very hairy cases this can take multiple passes.
                    # can we optimize this, or code it more elegantly?
                    max_passes_left = 10
                    while max_passes_left > 0:
                        this_chunk_end = min([x.end for x in inputs.values()] + [this_chunk_end])
                        if len(set([x.end for x in inputs.values()])) <= 1:
                            break
                        for d in self.depends_on:
                            inputs[d], back_to_buffer = inputs[d].split(
                                t=this_chunk_end, allow_early_split=True
                            )
                            self.input_buffer[d] = strax.Chunk.concatenate(
                                [back_to_buffer, self.input_buffer[d]]
                            )
                        max_passes_left -= 1
                    else:
                        raise RuntimeError(
                            f"{self} was unable to get time-consistent "
                            f"inputs after ten passess. Inputs: \n{inputs}\n"
                            f"Input buffer:\n{self.input_buffer}"
                        )

                    # Merge inputs of the same kind
                    inputs_merged = {
                        kind: strax.Chunk.merge([inputs[d] for d in deps_of_kind])
                        for kind, deps_of_kind in self.dependencies_by_kind().items()
                    }

                # Submit the computation
                # print(f"{self} calling with {inputs_merged}")
                if self.parallel and executor is not None:
                    new_future = executor.submit(self.do_compute, chunk_i=chunk_i, **inputs_merged)
                    pending_futures.append(new_future)
                    pending_futures = [f for f in pending_futures if not f.done()]
                    yield new_future
                else:
                    yield from self._iter_compute(chunk_i=chunk_i, **inputs_merged)

        except IterDone:
            # Check all sources are exhausted.
            # This is more than a check though -- it ensure the content of
            # all sources are requested all the way (including the final
            # Stopiteration), as required by lazy-mode processing requires
            for d in iters.keys():
                if self._fetch_chunk(d, iters):
                    raise RuntimeError(f"Plugin {d} terminated without fetching last {d}!")

            # This can happen especially in time range selections
            if hasattr(self.save_when, "values"):
                save_when = max([int(save_when) for save_when in self.save_when.values()])
            else:
                save_when = self.save_when
            if save_when > strax.SaveWhen.EXPLICIT:
                for d, buffer in self.input_buffer.items():
                    # Check the input buffer is empty
                    if buffer is not None and len(buffer):
                        raise RuntimeError(f"Plugin {d} terminated with leftover {d}: {buffer}")

        finally:
            self.cleanup(wait_for=pending_futures)

    def _iter_compute(self, chunk_i, **inputs_merged):
        """Either yields or returns strax chunks from the input."""
        yield self.do_compute(chunk_i=chunk_i, **inputs_merged)

    def cleanup(self, wait_for):
        pass
        # A standard plugin doesn't need to do anything here

    def _check_dtype(self, x, d=None):
        # There is an additional 'last resort' data type check
        # in the chunk initialization.
        # This one is broader and gives a more context-aware message.
        if d is None:
            assert not self.multi_output
            d = self.provides[0]
        pname = self.__class__.__name__
        if not isinstance(x, np.ndarray):
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver data type {d} as promised.\nDelivered a {type(x)}"
            )

        expect = strax.remove_titles_from_dtype(self.dtype_for(d))
        if not isinstance(expect, np.dtype):
            raise ValueError(f"Plugin {pname} expects {expect} as dtype??")
        got = strax.remove_titles_from_dtype(x.dtype)
        if got != expect:
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver "
                f"data type {d} as promised.\n"
                f"Promised:  {expect}\n"
                f"Delivered: {got}."
            )

    def do_compute(self, chunk_i=None, **kwargs):
        """Wrapper for the user-defined compute method.

        This is the 'job' that gets executed in different processes/threads during multiprocessing

        """
        for k, v in kwargs.items():
            if not isinstance(v, strax.Chunk):
                raise RuntimeError(
                    f"do_compute of {self.__class__.__name__} got a {type(v)} "
                    f"instead of a strax Chunk for {k}"
                )

        if len(kwargs):
            # Check inputs describe the same time range
            tranges = {k: (v.start, v.end) for k, v in kwargs.items()}
            start, end = list(tranges.values())[0]

            # For non-saving plugins, don't be strict, just take whatever
            # endtimes are available and don't check time-consistency
            # Side mark this wont work for a plugin which has a SaveWhen.NEVER and other
            # SaveWhen type.
            if hasattr(self.save_when, "values"):
                save_when = max([int(save_when) for save_when in self.save_when.values()])
            else:
                save_when = self.save_when
            if save_when <= strax.SaveWhen.EXPLICIT:
                # </start>This warning/check will be deleted, see UserWarning
                if len(set(tranges.values())) != 1:
                    start = min([v.start for v in kwargs.values()])
                    end = max([v.end for v in kwargs.values()])  # Don't delete
                    message = (
                        "New feature, we are ignoring inconsistent the "
                        "possible ValueError in time ranges for "
                        f"{self.__class__.__name__} of inputs: {tranges}"
                        "because this occurred in a save_when.NEVER "
                        "plugin. Report any findings in "
                        "github.com/AxFoundation/strax/issues/247"
                    )
                    warn(message, UserWarning)
                # This block will be deleted </end>
            elif len(set(tranges.values())) != 1:
                message = (
                    f"{self.__class__.__name__} got inconsistent time ranges of inputs: {tranges}"
                )
                raise ValueError(message)
        else:
            # This plugin starts from scratch
            start, end = None, None

        kwargs = {k: v.data for k, v in kwargs.items()}
        if self.compute_takes_chunk_i:
            kwargs["chunk_i"] = chunk_i
        if self.compute_takes_start_end:
            kwargs["start"] = start
            kwargs["end"] = end
        result = self.compute(**kwargs)

        return self._fix_output(result, start, end)

    def _fix_output(self, result, start, end, _dtype=None):
        if self.multi_output and _dtype is None:
            if not isinstance(result, dict):
                raise ValueError(
                    f"{self.__class__.__name__} is multi-output and should "
                    "provide a dict output {dtypename: result}"
                )
            return {d: self._fix_output(result[d], start, end, _dtype=d) for d in self.provides}

        if _dtype is None:
            assert not self.multi_output
            _dtype = self.provides[0]

        if not isinstance(result, strax.Chunk):
            if start is None:
                assert len(self.depends_on) == 0
                raise ValueError(
                    "Plugins without dependencies must return full strax "
                    f"Chunks, but {self.__class__.__name__} produced a "
                    f"{type(result)}!"
                )
            if isinstance(result, dict) and len(result) == 1:
                raise ValueError(
                    "Ran into single key results dict with key: "
                    f"{list(result.keys())}, cannot convert this to array of "
                    f"dtype {self.dtype_for(_dtype)}.\nSee "
                    "github.com/AxFoundation/strax/issues/238 for more info"
                )
            result = strax.dict_to_rec(result, dtype=self.dtype_for(_dtype))
            self._check_dtype(result, _dtype)
            result = self.chunk(start=start, end=end, data_type=_dtype, data=result)
        return result

    def chunk(self, *, start, end, data, data_type=None, run_id=None):
        if data_type is None:
            if self.multi_output:
                raise ValueError(
                    "Must give data_type when making chunks from a multi-output plugin"
                )
            data_type = self.provides[0]
        if run_id is None:
            run_id = self.run_id
        return strax.Chunk(
            start=start,
            end=end,
            run_id=run_id,
            data_kind=self.data_kind_for(data_type),
            data_type=data_type,
            dtype=self.dtype_for(data_type),
            data=data,
            target_size_mb=self.chunk_target_size_mb,
        )

    def compute(self, **kwargs):
        raise NotImplementedError
