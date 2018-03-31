"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from functools import partial
import inspect
import os
import re

import numpy as np
import pandas as pd

import strax
import strax.chunk_arrays as ca

__all__ = ('register_plugin provider data_info '
           'StraxPlugin MergePlugin LoopPlugin').split()


##
# Plugin registry
# This global dict tracks which plugin provides which data
##

REGISTRY = dict()


def register_plugin(plugin_class, provides=None):
    """Register plugin_class as provider for plugin_class.provides and
    other data types listed in provides.
    :param plugin_class: class inheriting from StraxPlugin
    :param provides: list of additional data types which this plugin provides.
    """
    if provides is None:
        provides = []
    global REGISTRY
    inst = plugin_class()
    for p in [inst.provides] + provides:
        REGISTRY[p] = inst
    return plugin_class


def provider(data_name):
    """Return instance of plugin that provides data_name"""
    try:
        return REGISTRY[data_name]
    except KeyError:
        raise KeyError(f"No plugin registered that provides {data_name}")


def data_info(data_name):
    """Return pandas DataFrame describing fields in data_name"""
    p = provider(data_name)
    display_headers = ['Field name', 'Data type', 'Comment']
    result = []
    for name, dtype in strax.utils.unpack_dtype(p.dtype):
        if isinstance(name, tuple):
            title, name = name
        else:
            title = ''
        result.append([name, dtype, title])
    return pd.DataFrame(result, columns=display_headers)


##
# Base plugin
##

class StraxPlugin:
    data_kind: str
    depends_on: tuple
    provides: str
    compressor: str = 'blosc'       # Compressor to use for files

    def __init__(self):
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

    def get(self, data_dir):
        """Iterate over results from data_dir. If they do not exist,
        process them."""
        out_dir = os.path.join(data_dir, self.provides)
        if os.path.exists(out_dir):
            print(f"{self.provides} already exists, yielding")
            yield from strax.io_chunked.read_chunks(out_dir)
        else:
            print(f"{self.provides} does not exist, processing")
            yield from self.iter(data_dir)

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

    def iter(self, data_dir, n_per_iter=None):
        """Yield result chunks for processing input_dir
        :param n_per_iter: pass at most this many rows to compute
        """
        deps_by_kind = self.dependencies_by_kind()
        iters = {d: provider(d).get(data_dir)
                 for d in self.depends_on}

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
            iters.update(ca.sync_iters(
                ca.same_length,
                {d: iters[d] for d in deps}))

        while True:
            try:
                yield self.compute(**{d: next(iters[d])
                                      for d in self.depends_on})
            except StopIteration:
                return

    def process_and_slurp(self, input_dir, **kwargs):
        """Return results for processing data_dir"""
        return np.concatenate(list(self.iter(input_dir, **kwargs)))

    def save(self, input_dir, output_dir=None, chunk_size=int(5e7), **kwargs):
        """Process data_dir and save the results there"""
        if output_dir is None:
            output_dir = input_dir
        out_dir = os.path.join(output_dir, self.provides)

        it = self.iter(input_dir, **kwargs)
        it = strax.chunk_arrays.fixed_size_chunks(it, chunk_size)
        strax.io_chunked.save_to_dir(it, out_dir, compressor=self.compressor)

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
