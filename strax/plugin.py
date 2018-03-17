import inspect
import re
import os

import numpy as np

import strax

__all__ = 'register_plugin StraxPlugin MergePlugin'.split()


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

    def iter(self, input_dir, pacemaker=None, pbar=True):
        """Yield result chunks for processing input_dir
        """
        # Which dependency decides the chunking?
        if pacemaker is None:
            if self.chunking in self.depends_on:
                # We have to chunk output like one of the dependencies,
                # so let's use that one and save the Saver some trouble
                pacemaker = self.chunking
            else:
                pacemaker = self.depends_on[0]
        other_deps = [k for k in self.depends_on if k != pacemaker]

        dirnames = {k: os.path.join(input_dir, k) for k in self.depends_on}
        chunk_iters = {k: strax.io_chunked.read_chunks(dn)
                       for k, dn in dirnames.items()}
        my_iter = chunk_iters[pacemaker]

        # Add a progress bar if we have tqdm installed and pbar=True
        if pbar:
            try:
                from tqdm import tqdm         # noqa
                n_chunks = len(strax.io_chunked.chunk_files(
                    dirnames[pacemaker]))
                my_iter = tqdm(my_iter,
                             total=n_chunks,
                             desc=f"Computing {self.provides} of {input_dir}")
            except ImportError:
                pass

        buffers = {k: next(chunk_iters[k]) for k in other_deps}
        for x in my_iter:
            dep_data = {pacemaker: x}
            for k in other_deps:

                while len(buffers[k]) < len(x):
                    buffers[k] = np.concatenate((buffers[k],
                                                 next(chunk_iters[k])))

                dep_data[k] = buffers[k][:len(x)]
                buffers[k] = buffers[k][len(x):]

            yield self.compute(**dep_data)

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

class MergePlugin(StraxPlugin):
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
