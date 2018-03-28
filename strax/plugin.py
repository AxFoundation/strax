"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
import inspect
import re
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import strax
from collections import OrderedDict
from strax import chunk_arrays

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
    return REGISTRY[data_name]


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

    def iter(self, input_dir, pbar=True, n_per_iter=None):
        """Yield result chunks for processing input_dir
        """
        # deps_of_kind = {kind: [dep, dep, ..], ...}
        # kind_of provides reverse lookup: dep -> kind
        deps_of_kind = OrderedDict()
        kind_of = dict()
        for d in self.depends_on:
            kind_of[d] = k = provider(d).data_kind
            deps_of_kind.setdefault(k, [])
            deps_of_kind[k].append(d)

        # At least one dependency of each kind should have time information.
        # (which we need to sync among different data kinds)
        # We'll call this the "key dependency" of that type.
        # key_for = {kind: key_dep, ...}
        key_for = OrderedDict()
        for k, ds in deps_of_kind.items():
            for d in ds:
                if 'time' in provider(d).dtype.names:
                    key_for[k] = d
            if k not in key_for:
                raise ValueError(f"One of the dependencies {ds} of the kind "
                                 f"{k} must provide time information!")

        # Key dependency of first datatype becomes the master iterator
        master = key_for[list(deps_of_kind.keys())[0]]

        # Grab iterators/pacers for each dependency
        pacers = dict()
        master_iter = None
        for d in self.depends_on:
            dn = os.path.join(input_dir, d)
            it = strax.io_chunked.read_chunks(dn)
            if d != master:
                pacers[d] = chunk_arrays.ChunkPacer(it)
                continue

            master_iter = it
            if pbar:
                n_chunks = len(strax.io_chunked.chunk_files(dn))
                desc = f"Computing {self.provides} of {input_dir}"
                master_iter = iter(tqdm(master_iter,
                                        total=n_chunks,
                                        desc=desc))
            if n_per_iter is not None:
                master_iter = chunk_arrays.fixed_length_chunks(master_iter,
                                                               n_per_iter)

        for i, x in enumerate(master_iter):
            result = {master: x}

            # Add key dependencies
            for d in key_for.values():
                if d == master:
                    continue
                # TODO: assumed sorted on endtime: only true if nonoverlapping!
                result[d] = pacers[d].get_until(strax.endtime(x[-1]),
                                                f=strax.endtime)

            # Add rest now that syncing is known
            for d in self.depends_on:
                if d in key_for.values():
                    continue
                key_dep = key_for[kind_of[d]]
                result[d] = pacers[d].get_n(len(result[key_dep]))

            yield self.compute(**result)

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
