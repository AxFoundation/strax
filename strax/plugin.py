import inspect
import re
import os

import numpy as np

import strax


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

    def iter(self, data_dir):
        """Yield result chunks for processing data_dir"""
        # Get iterator over dependency chunks
        # i.e. something that gives {'dep1': array, 'dep2': array} on each iter
        dep_chunks = {k: strax.io_chunked.read_chunks(
                        os.path.join(data_dir, k))
                      for k in self.depends_on}
        dep_chunks_tr = (dict(zip(dep_chunks, col))
                         for col in zip(*dep_chunks.values()))

        # TODO: progress bar ETA: need to fetch total n_chunks
        for ch in dep_chunks_tr:
            yield self.compute(**ch)

    def process_and_slurp(self, data_dir):
        """Return results for processing data_dir"""
        return np.concatenate(list(self.iter(data_dir)))

    def _get_saver(self, data_dir):
        out_dir = os.path.join(data_dir, self.provides)
        if self.chunking in self.depends_on:
            # Chunk like our dependencies
            return strax.io_chunked.Saver(out_dir)
        else:
            # Make our own chunks
            return strax.io_chunked.ThresholdSizeSaver(out_dir)

    def process_and_save(self, data_dir):
        """Process data_dir and save the results there"""
        with self._get_saver(data_dir) as saver:
            for out in self.iter(data_dir):
                saver.feed(out)

    def iter_and_save(self, data_dir):
        with self._get_saver(data_dir) as saver:
            for out in self.iter(data_dir):
                saver.feed(out)
                yield out

    def compute(self, **kwargs):
        raise NotImplementedError


def camel_to_snake(x):
    """Convert x from CamelCase to snake_case"""
    # From https://stackoverflow.com/questions/1175208
    x = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', x)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', x).lower()
