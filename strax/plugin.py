"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
import inspect
import re
import os
from tqdm import tqdm

import numpy as np

import strax

__all__ = 'register_plugin StraxPlugin MergePlugin LoopPlugin'.split()


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

##
# These should go to utils
##
import numba





##
# Base plugin
##

class StraxPlugin:
    data_kind: str
    depends_on: tuple
    provides: str
    chunking: str

    def __init__(self):
        self.dtype = np.dtype(self.dtype)

        if not hasattr(self, 'depends_on'):
            # Infer dependencies from 'process' argument names
            process_params = inspect.signature(self.compute).parameters.keys()
            self.depends_on = tuple(process_params)

        if not hasattr(self, 'data_kind'):
            # Assume data kind is the same as the first dependency
            self.data_kind = REGISTRY[self.depends_on[0]].data_kind

        if not hasattr(self, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = camel_to_snake(self.__class__.__name__)
            self.provides = snake_name

        if not hasattr(self, 'chunking'):
            # No chunking scheme specified: start a new one
            self.chunking = self.provides

    def iter(self, input_dir, pbar=True, n_per_iter=None):
        """Yield result chunks for processing input_dir
        """
        # Which dependency decides the chunking? Call this the 'pacemaker'
        if self.chunking in self.depends_on:
            # We have to chunk output like one of the dependencies
            if n_per_iter:
                raise ValueError("Will get into trouble, saver just passes on")
            pacemaker = self.chunking
        else:
            pacemaker = self.depends_on[0]

        dep_plugins = {k: REGISTRY[k] for k in self.depends_on}
        data_kinds = list(set([p.data_kind
                              for p in dep_plugins.values()]))
        multi_kind = len(data_kinds) > 1

        if multi_kind:
            # We depend on several data kinds (e.g. peaks and events).
            # At least one dependency of each kind should have time information
            # necessary to chunk consistently.
            # We'll call these "key dependencies".
            key_of = dict()
            for k in data_kinds:
                deps_this_kind = [depname
                                  for depname, p in dep_plugins.items()
                                  if p.data_kind == k]
                for d in deps_this_kind:
                    if 'time' in dep_plugins[d].dtype.names:
                        key_of[k] = d
                if k not in key_of:
                    raise ValueError("One of the dependencies "
                                     f"{deps_this_kind} of the kind {k} "
                                     "must provide time information!")

            key_deps = list(key_of.values())
            if pacemaker not in key_deps:
                raise ValueError(f"Pacemaker {pacemaker} should have been a "
                                 "dependency that provides time info (like "
                                 f"{key_deps}.")
            other_deps = [k for k in self.depends_on
                          if k not in key_deps]

        else:
            key_of = {dep_plugins[pacemaker].data_kind: pacemaker}
            key_deps = []
            other_deps = [k for k in self.depends_on
                          if k != pacemaker]

        # Get iterators over chunk files for each dependency
        dirnames = {k: os.path.join(input_dir, k) for k in self.depends_on}
        chunk_iters = {k: strax.io_chunked.read_chunks(dn)
                       for k, dn in dirnames.items()}
        pacemaker_iter = chunk_iters[pacemaker]

        if pbar:
            # Add progress bar
            n_chunks = len(strax.io_chunked.chunk_files(
                dirnames[pacemaker]))
            desc = f"Computing {self.provides} of {input_dir}"
            pacemaker_iter = tqdm(pacemaker_iter, total=n_chunks, desc=desc)

        if n_per_iter is not None:
            # Use a maximum chunk size for processing
            def _make_my_iter():
                for y in pacemaker_iter:
                    while len(y) > n_per_iter:
                        # print("Emmiting constrained chunk")
                        res, y = np.split(y, [n_per_iter])
                        yield res
                    if len(y):
                        # print("Emmiting tail chunk")
                        yield y
            my_iter = _make_my_iter()
        else:
            # Process whole chunks from disk at once
            my_iter = pacemaker_iter

        buffers = {k: next(chunk_iters[k])
                   for k in self.depends_on if k != pacemaker}
        for x in my_iter:
            dep_data = {pacemaker: x}

            if multi_kind:
                # Grab the right chunks of the key dependencies
                last_endtime = strax.endtime(x[-1])

                for k in key_deps:
                    if k == pacemaker:
                        continue
                    b = buffers[k]

                    # Assemble enough data
                    while True:
                        last = strax.endtime(b[-1])
                        if last > last_endtime:
                            break
                        try:
                            new = next(chunk_iters[k])
                            b = np.concatenate((b, new))
                        except StopIteration:
                            break

                    n = strax.first_index_beyond(strax.endtime(b),
                                                 last_endtime)
                    dep_data[k], buffers[k] = np.split(b, [n])

            # Grab right chunks of other dependencies (which probably do not
            # have time information, like the key dependencies)
            for k in other_deps:
                data_kind = dep_plugins[k].data_kind
                key_dep = key_of[data_kind]
                n = len(dep_data[key_dep])
                b = buffers[k]

                while len(b) < n:
                    b = np.concatenate((b, next(chunk_iters[k])))

                dep_data[k], buffers[k] = np.split(b, [n])

            yield self.compute(**dep_data)

    def process_and_slurp(self, input_dir, **kwargs):
        """Return results for processing data_dir"""
        return np.concatenate(list(self.iter(input_dir, **kwargs)))

    def _saver(self, output_dir):
        out_dir = os.path.join(output_dir, self.provides)
        if self.chunking in self.depends_on:
            # Chunk like our dependencies (taken care of during iter)
            return strax.io_chunked.Saver(out_dir)
        else:
            # Make fixed-size chunks
            return strax.io_chunked.ThresholdSizeSaver(out_dir)

    def save(self, input_dir, output_dir=None, **kwargs):
        """Process data_dir and save the results there"""
        if output_dir is None:
            output_dir = input_dir
        with self._saver(output_dir) as saver:
            for out in self.iter(input_dir, **kwargs):
                saver.feed(out)

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
        data_kind = {d: REGISTRY[d].data_kind
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
                base,
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

        ps = [REGISTRY[x] for x in self.depends_on]

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
