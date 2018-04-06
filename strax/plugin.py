"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from functools import partial
from itertools import count
import inspect
import re

import numpy as np

import strax
import strax.chunk_arrays as ca
from strax.core import provider, register_plugin

from strax.utils import exporter
export, __all__ = exporter()


@export
class StraxPlugin:
    __version__: str
    data_kind: str
    depends_on: tuple
    provides: str
    compressor: str = 'blosc'       # Compressor to use for files

    def __init__(self):
        self.log = strax.setup_logger(self.__class__.__name__)
        self.dtype = np.dtype(self.dtype)

        if not hasattr(self, 'depends_on'):
            # Infer dependencies from self.compute's argument names
            process_params = inspect.signature(self.compute).parameters.keys()
            process_params = [p for p in process_params if p != 'kwargs']
            self.depends_on = tuple(process_params)

        if not hasattr(self, 'data_kind'):
            # Assume data kind is the same as the first dependency
            self.data_kind = provider(self.depends_on[0]).data_kind

        if not hasattr(self, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = camel_to_snake(self.__class__.__name__)
            self.provides = snake_name

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

    def dependencies_by_kind(self, require_time=True):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True (default), one dependency of each kind
        must provide time information. It will be put first in the list.
        """
        deps_by_kind = dict()
        key_deps = []
        for d in self.depends_on:
            p = provider(d)

            k = p.data_kind
            deps_by_kind.setdefault(k, [])

            # If this has time information, put it first in the list
            if require_time and 'time' in p.dtype.names:
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

    def iter(self, iters, output_mailbox, n_per_iter=None):
        """Yield result chunks for processing input_dir
        :param iters: dict with iterators over dependencies
        :param output_mailbox: mailbox to send results to
        :param n_per_iter: pass at most this many rows to compute
        """
        deps_by_kind = self.dependencies_by_kind()

        if n_per_iter is not None:
            # Apply additional flow control
            for kind, deps in deps_by_kind.items():
                d = deps[0]
                iters[d] = ca.fixed_length_chunks(iters[d], n=n_per_iter)
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

        for result_i in count():
            try:
                compute_kwargs = {d: next(iters[d])
                                  for d in self.depends_on}
            except StopIteration:
                output_mailbox.close()
                return
            # We might punt the compute to a processpool in the future
            r = self.compute(**compute_kwargs)
            output_mailbox.send(r, number=result_i)

    def compute(self, **kwargs):
        raise NotImplementedError


def camel_to_snake(x):
    """Convert x from CamelCase to snake_case"""
    # From https://stackoverflow.com/questions/1175208
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()


##
# Special plugins
##

@export
class LoopPlugin(StraxPlugin):
    """Plugin that disguises multi-kind data-iteration by an event loop
    """

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for LoopPlugin')

        # Data kind to look over is set by first dependency
        self.loop_over = provider(self.depends_on[0]).data_kind

        super().__init__()

    def compute(self, **kwargs):
        # Merge data of each data kind
        deps_by_kind = self.dependencies_by_kind()
        things_by_kind = {
            k: strax.merge_arrs([kwargs[d] for d in deps])
            for k, deps in deps_by_kind.items()
        }

        # Group into lists of things (e.g. peaks)
        # contained in the base things (e.g. events)
        base = things_by_kind[self.loop_over]
        for k, things in things_by_kind.items():
            if k != self.loop_over:
                things_by_kind[k] = strax.split_by_containment(things, base)

        results = np.zeros(len(base), dtype=self.dtype)
        for i in range(len(base)):
            r = self.compute_loop(base[i],
                                  **{k: things_by_kind[k][i]
                                     for k in deps_by_kind
                                     if k != self.loop_over})

            # Convert from dict to array row:
            for k, v in r.items():
                results[i][k] = v

        return results

    def compute_loop(self, base, **kwargs):
        raise ValueError


@export
class MergePlugin(StraxPlugin):
    """Plugin that merges data from its dependencies
    """

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for MergePlugin')

        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergePlugins can only merge data of the same "
                             "kind, but got multiple kinds: "
                             + str(deps_by_kind))

        for k in deps_by_kind:
            self.data_kind = k
            # No break needed, there's just one item

        self.dtype = sum([strax.unpack_dtype(provider(d).dtype)
                          for d in self.depends_on], [])

        super().__init__()

    def compute(self, **kwargs):
        return strax.merge_arrs(list(kwargs.values()))


@export
class PlaceholderPlugin(StraxPlugin):
    """Plugin that throws NotImplementedError when asked to compute anything"""
    depends_on = tuple()

    def compute(self):
        raise NotImplementedError("No plugin registered that "
                                  f"provides {self.provides}")


@register_plugin
class Records(PlaceholderPlugin):
    """Placeholder plugin for something (e.g. a DAQ or simulator) that
    provides strax records.
    """
    data_kind = 'records'
    dtype = strax.record_dtype()
