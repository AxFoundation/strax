import inspect
import re
import os

import numpy as np

import strax

__all__ = ('StraxPlugin', 'register_plugin')


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
# Base plugin
##

class StraxPlugin:
    depends_on: tuple
    provides: str
    chunking: str

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            # Infer dependencies from 'process' argument names
            process_params = inspect.signature(self.compute).parameters.keys()
            self.depends_on = tuple(process_params)

        if not hasattr(self, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = camel_to_snake(self.__class__.__name__)
            self.provides = snake_name

        if not hasattr(self, 'chunking'):
            # No chunking scheme specified: start a new one
            self.chunking = self.provides

    def iter(self, input_dir, pbar=True):
        """Yield result chunks for processing input_dir"""
        # Get iterator over dependency chunks
        # i.e. something that gives {'dep1': array, 'dep2': array} on each iter
        dep_dirs = [os.path.join(input_dir, k) for k in self.depends_on]
        dep_chunk_iters = {k: strax.io_chunked.read_chunks(dirname)
                           for k, dirname in zip(self.depends_on, dep_dirs)}
        my_it = (dict(zip(dep_chunk_iters, col))
                 for col in zip(*dep_chunk_iters.values()))

        if pbar:
            # Make a progress bar if we have tqdm installed
            try:
                from tqdm import tqdm         # noqa
                n_chunks = len(strax.io_chunked.chunk_files(dep_dirs[0]))
                my_it = tqdm(my_it,
                             total=n_chunks,
                             desc=f"Computing {self.provides} of {input_dir}")
            except ImportError:
                pass

        for x in my_it:
            yield self.compute(**x)

    def process_and_slurp(self, input_dir, **kwargs):
        """Return results for processing data_dir"""
        return np.concatenate(list(self.iter(input_dir, **kwargs)))

    def _saver(self, output_dir):
        out_dir = os.path.join(output_dir, self.provides)
        if self.chunking in self.depends_on:
            # Chunk like our dependencies
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
# Specialized plugins
##

class MergePlugin(strax.StraxPlugin):
    """Plugin that merges data from its dependencies
    """

    def __init__(self):
        # TODO: check data type is the same
        # TODO: check chunking is the same (until multi-chunk looping properly
        # supported)
        self.dtype = sum([REGISTRY[x].dtype
                          for x in self.depends_on],
                         [])
        super().__init__()

    def compute(self, **kwargs):
        return strax.merge_arrs(list(kwargs.values()))
