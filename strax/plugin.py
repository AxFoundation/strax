"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
import inspect
from functools import partial
import re
import os

import numpy as np
import pandas as pd

import strax
from strax.chunk_arrays import sync_iters, same_length, same_stop

__all__ = ('register_plugin provider data_info '
           'StraxPlugin MergePlugin LoopPlugin').split()


##
# Plugin registry
# This global dict tracks which plugin provides which data
##

REGISTRY = dict()


def register_plugin(plugin_class):
    global REGISTRY
    inst = plugin_class()
    REGISTRY[inst.provides] = inst
    return plugin_class


def provider(data_name):
    try:
        return REGISTRY[data_name]
    except KeyError:
        raise KeyError(f"No plugin registered that provides {data_name}")


def data_info(data_name):
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
    compressor: str = 'blosc'

    def __init__(self):
        self.dtype = np.dtype(self.dtype)

        if not hasattr(self, 'depends_on'):
            # Infer dependencies from 'process' argument names
            process_params = inspect.signature(self.compute).parameters.keys()
            self.depends_on = tuple(process_params)

        if not hasattr(self, 'data_kind'):
            # Assume data kind is the same as the first dependency
            self.data_kind = provider(self.depends_on[0]).data_kind

        if not hasattr(self, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = camel_to_snake(self.__class__.__name__)
            self.provides = snake_name

    def get(self, data_dir):
        out_dir = os.path.join(data_dir, self.provides)
        if os.path.exists(out_dir):
            print(f"{self.provides} already exists, yielding")
            yield from strax.io_chunked.read_chunks(out_dir)
        else:
            print(f"{self.provides} does not exist, processing")
            yield from self.iter(data_dir)

    def iter(self, data_dir, pbar=True, n_per_iter=None):
        """Yield result chunks for processing input_dir
        """
        # Get iterators over the dependencies
        # For each data kind, identify a 'key dependency' that contains
        # time information. We use this to sync the iteration over all
        # dependencies of that data kind.
        iters = dict()
        deps_by_kind = dict()
        key_deps = []
        for d in self.depends_on:
            p = provider(d)
            iters[d] = p.get(data_dir)

            k = p.data_kind
            deps_by_kind.setdefault(k, [])

            # If this has time information, put it first in the list
            # so we can use it for syncing the others
            if 'time' in p.dtype.names:
                key_deps.append(d)
                deps_by_kind[k].insert(0, d)
            else:
                deps_by_kind[k].append(d)

        for k, d in deps_by_kind.items():
            if not d[0] in key_deps:
                raise ValueError("missing key dep for %s" % k)

        # Sync the iterators of the key dependencies by endtime
        iters.update(sync_iters(partial(same_stop, func=strax.endtime),
                                {d[0]: iters[d[0]]
                                 for d in deps_by_kind.values()}))

        # Sync the remaining iterators to the key dependencies
        for deps in deps_by_kind.values():
            iters.update(sync_iters(same_length,
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
    loop_over: str

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on is mandatory for LoopPlugin')
        super().__init__()
        if not hasattr(self, 'loop_over'):
            self.loop_over = self.data_kind

    def compute(self, **kwargs):
        # Merge arguments of each data kind
        data_kind = {d: provider(d).data_kind
                     for d in self.depends_on}
        merged = dict()
        all_kinds = set(list(data_kind.values()))
        for k in all_kinds:
            to_merge = []
            for d in self.depends_on:
                if data_kind[d] == k:
                    to_merge.append(kwargs[d])
            merged[k] = strax.merge_arrs(to_merge)

        # Which base thing (e.g. event)
        # do the other things (e.g. peaks) belong to?
        base = merged[self.loop_over]
        which_base = {k: strax.fully_contained_in(merged[k], base)
                      for k in all_kinds if k != self.loop_over}

        results = np.zeros(len(base), dtype=self.dtype)
        for i in range(len(base)):
            r = self.compute_loop(
                base[i],
                **{k: merged[k][which_base[k] == i]
                   for k in all_kinds if k != self.loop_over})

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

        ps = [provider(x) for x in self.depends_on]

        data_kinds = set([p.data_kind for p in ps])
        if len(data_kinds) > 1:
            raise ValueError("MergePlugins can only merge data of the same "
                             "kind, but got multiple kinds: {data_kinds}")
        self.data_kind = self.depends_on[0]

        self.dtype = sum([strax.unpack_dtype(p.dtype) for p in ps], [])

        super().__init__()

    def compute(self, **kwargs):
        return strax.merge_arrs(list(kwargs.values()))


class PlaceholderPlugin(StraxPlugin):
    """Plugin that throws NotImplementedError when asked to compute anything"""

    def compute(self, **kwargs):
        raise NotImplementedError("No plugin registered that "
                                  "provides {self.provides}")


@register_plugin
class Records(PlaceholderPlugin):
    """Placeholder plugin for something (e.g. a DAQ or simulator) that
    provides strax records.
    """
    data_kind = 'records'
    dtype = strax.record_dtype()
