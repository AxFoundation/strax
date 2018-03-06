import numpy as np
import dask

import strax


class StraxExtension:
    __version__: str
    depends_on: tuple = ()    # Auto-filled from compute kwargs?
    provides: str
    store: False

    # Instance attributes that hold state temporarily
    run_name: str
    lineage: dict

    def version(self, run_id=None):
        """Return version number applicable to the run_id.
        Most extensions just have a single version (in __version__),
        but some may be at different versions for different runs
        (e.g. time-dependent corrections).
        """
        return self.__version__

    def task_key(self, run_id):
        """Return key identifying computation tasks in the task graph
        This is usually (ClassName, versionstring)
        """
        return self.__class__.__name__, self.version(run_id)

    def task_graph(self, run_id, provides):
        """Return the taskgraph for computing this extension for run_id.

        The task graph follows the dask specification.
        Keys are provided by the 'key' method of each extension,
        the values are the task tuples:
         - the computation task (_compute method of extension to run)
         - run_id
         - lineage   --  set of keys that this extension depends on
         - *dependencies -- any further arguments to compute
        """
        # Which extensions are providing the dependencies?
        depends_on = [provides[k] for k in self.depends_on]
        keys_of_deps = [k.task_key(run_id) for k in depends_on]

        # Get and merge their task graphs
        d = {}
        for k in depends_on:
            d.update(k.task_graph(run_id))

        # Compute lineage of this extension
        lineage = set(keys_of_deps)
        for k in keys_of_deps:
            lineage |= d[k][2]

        # Get the task to compute this extension, and add it to the task graph
        task = (self._compute,
                run_id,
                lineage,
                *[k.task_key(run_id) for k in depends_on])
        d[self.task_key(run_id)] = task
        return d

    def _compute(self, run_id, lineage, *dependencies):
        # Store run_id and lineage in attributes
        # so user-defined compute-function does not need to take them
        self.run_id = run_id
        self.lineage = lineage

        result = self.compute(*dependencies)

        if self.store:
            key_for_cache = (run_id, self.task_key(run_id), lineage)
            CACHE.save(key_for_cache, self.save_to_cache, result)

            filename = CACHE.request_new_filename(*key_for_cache)
            self.save_to_cache(filename, result)
            CACHE.register_file(filename, *key_for_cache)

        return result

    def compute(self, *dependencies):
        raise NotImplementedError

    @staticmethod
    def save_to_cache(result, filename):
        strax.save(filename, result)

    @staticmethod
    def load_from_cache(filename):
        return strax.load(filename)


##
# Example extensions
##

# ADC -> PE conversion factors
# This should go to some configuration file!!
to_pe = 1e-3 * np.array(
    [7.05, 0.0, 0.0, 8.09, 4.38, 7.87, 3.58, 7.5, 7.44, 4.82, 7.07, 5.79,
     0.0, 5.55, 7.95, 7.02, 6.39, 8.1, 7.15, 7.43, 7.15, 11.4, 3.97, 7.28,
     5.41, 7.4, 0.0, 0.0, 7.04, 7.27, 4.22, 16.79, 4.14, 7.04, 0.0, 5.38,
     7.39, 7.02, 4.53, 5.17, 7.13, 5.48, 4.6, 7.33, 6.14, 6.52, 7.59,
     4.76, 7.56, 7.54, 4.57, 4.6, 7.12, 8.0, 4.7, 8.68, 3.74, 4.97, 10.36,
     7.53, 6.02, 12.45, 0.0, 4.49, 4.82, 0.0, 8.13, 7.27, 3.55, 5.65,
     4.55, 8.64, 7.97, 0.0, 3.57, 3.69, 5.87, 5.12, 9.8, 0.0, 5.08, 4.09,
     3.87, 8.17, 6.73, 9.03, 0.0, 6.93, 0.0, 6.52, 7.39, 0.0, 4.92, 7.48,
     5.82, 4.05, 3.9, 5.77, 8.14, 7.62, 7.61, 5.55, 0.0, 7.12, 5.02, 4.57,
     4.46, 7.44, 3.57, 7.58, 7.16, 7.33, 7.69, 6.03, 5.87, 9.64, 4.68,
     7.88, 0.0, 10.84, 7.0, 3.62, 7.5, 7.45, 7.69, 7.69, 3.49, 3.61, 7.44,
     6.38, 0.0, 5.1, 3.72, 5.22, 0.0, 0.0, 4.43, 0.0, 3.87, 0.0, 3.6,
     5.35, 8.4, 5.1, 6.45, 5.07, 4.28, 3.5, 0.0, 7.28, 0.0, 4.25, 0.0,
     4.72, 6.26, 7.28, 5.34, 7.55, 3.85, 5.54, 7.5, 7.31, 0.0, 7.76, 7.57,
     6.66, 7.29, 0.0, 7.59, 3.8, 3.58, 5.21, 4.29, 7.36, 7.76, 4.0, 6.23,
     5.86, 0.0, 7.34, 3.58, 3.57, 5.26, 0.0, 7.67, 4.05, 4.3, 4.21, 7.59,
     7.59, 0.0, 6.41, 4.86, 3.73, 5.09, 7.59, 7.64, 7.7, 0.0, 5.25, 8.0,
     5.32, 7.91, 0.0, 4.41, 11.82, 0.0, 4.51, 7.05, 8.63, 5.12, 4.45,
     4.03, 0.0, 0.0, 3.54, 4.18, 9.5, 3.64, 3.67, 7.28, 3.59, 5.03, 3.6,
     5.4, 7.18, 3.73, 6.21, 6.47, 3.7, 7.69, 4.58, 7.46, 6.74, 0.0, 3.66,
     7.49, 7.55, 3.64, 0.0, 7.34, 4.06, 3.74, 3.97, 0.0, 4.29, 4.96, 3.77,
     8.57, 8.57, 8.57, 8.57, 8.57, 8.57, 214.29, 171.43, 171.43, 171.43,
     171.43, 171.43]
)
samples_per_record = 110
n_channels = len(to_pe)


class RawRecords(StraxExtension):
    __version__ = strax.__version__
    provides = 'raw_records'
    dtype = strax.record_dtype(samples_per_record)

    def compute(self):
        raw_records = strax.load(self.run_name)
        raw_records = strax.sort_by_time(raw_records)
        strax.baseline(raw_records)
        strax.integrate(raw_records)
        return raw_records


# TODO: support for providing multiple things? Something else splits?
class PeakDetails(StraxExtension):
    __version__ = strax.__version__
    store = 'always'

    provides = 'processed_records'
    depends_on = ('raw_records',)
    dtype = strax.record_dtype(samples_per_record)

    def compute(self, raw_records):
        processed_records = strax.exclude_tails(raw_records, to_pe)
        hits = strax.find_hits(processed_records)
        strax.cut_outside_hits(processed_records, hits)
        peak_details = strax.find_peaks(hits, to_pe,
                                        gap_threshold=300,
                                        min_hits=3)
        strax.sum_waveform(peak_details, processed_records, to_pe)
        return dict(peak_details=peak_details)


class PeakWidths(StraxExtension):
    __version__ = '0.1'
    provides = 'peak_widths'
    depends_on = ('peak_details',)
    dtype = np.dtype([('width_area_decile', (np.float32, 11)),
                      ('width_std', np.float32),
                      ('width_full', np.float32)])

    def compute(self, peak_details):
        d = np.zeros(len(peak_details), dtype=self.dtype)
        d['width_full'] = peak_details['length'] * peak_details['dt']
        return d


##
# Loading
##


def task_graph(run_id, extensions_to_load, who_provides=None):

    if who_provides is None:
        who_provides = dict()
    who_provides = {**WHO_PROVIDES, **who_provides}
    extensions_to_load = [who_provides[e] for e in extensions_to_load]
    final_targets = [e.task_key(run_id) for e in extensions_to_load]
    task_graph = dict()
    for e in extensions_to_load:
        task_graph.update(e.task_graph(run_id, who_provides))

    # Build a new task graph with computes replaced by load_from_cache
    # as far downstream as possible.
    new_graph = {}
    stack = final_targets.copy()
    while len(stack):
        key = stack.pop()       # key = (ExtensionClassName, version)
        if key in new_graph:
            continue
        old_task = task_graph[key]
        filename = CACHE.find(run_id, key, old_task[2])
        if filename:
            new_graph[key] = (who_provides[key[0]].load_from_cache, filename)
        else:
            new_graph[key] = old_task
            # Walk further upstream: check if we can load dependencies
            stack.extend(old_task[3:])

    # If we loaded some things from cache, many computations are now obsolete
    return dask.optimization.cull(new_graph, final_targets)[0]


##
# Cache
##

from hashlib import sha1
import os
import json



class FolderBasedCache:
    """Simple cache that stores everything in a single folder
    Results are placed in subfolders named by run_id,
    with files named named EXT_VERSION_HASH, where HASH identifies lineage
    """

    def __init__(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.folder = folder

    def request_new_filename(self, run_id, key, lineage):
        """Return filename where new result should be placed
        The file is not registered yet, call register_file after you've
        saved the file successfully.
        """
        return self._filename(run_id, key, lineage)

    def find(self, run_id, key, lineage):
        """Return filename for existing result of key,
        or None if no such exists"""
        filename = self._filename(run_id, key, lineage)
        if os.path.exists(filename):
            return filename

    def delete(self, run_id, key, lineage):
        filename = self.find(run_id, key, lineage)
        if filename is None:
            raise KeyError("Cannot delete, file does not exist")
        os.remove(filename)

    def register_file(self, filename, run_id, key, lineage):
        """Register filename as a result for key"""
        pass

    def _filename(self, run_id, key, lineage):
        # Create a hash of the json of the (key, lineage) tuple
        h = sha1(json.dumps((key, tuple(sorted(lineage.items())))))
        # Version string often has dots, which could be interpreted as
        # extension separators. I'll replace them with '-' and reserve '_'
        # for the separator between fields in the filename
        filename = '_'.join([str(x) for x in key] + [h]).replace('.', '-')
        return os.path.join(self.folder, run_id, filename)


# TODO: combinedcache cache type
# Track writeability. If you request a new filename, only returns one if it is
# actually writeable.
CACHE = FolderBasedCache('strax_data')



##
# Extension registry
##

WHO_PROVIDES = dict()

def register_extension(ext, result_name):
    global WHO_PROVIDES
    inst = ext()
    WHO_PROVIDES[ext.__class__.__name__] = inst
    WHO_PROVIDES[result_name] = inst


for ext, result_name in [(RawRecords, 'raw_records'),
                         (PeakDetails, 'peak_details'),
                         (PeakWidths, 'peak_widths')]:
    register_extension(ext, result_name)
