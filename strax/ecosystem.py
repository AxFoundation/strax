import numpy as np
import dask

import strax

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


class StraxExtension:
    __version__: str
    depends_on: tuple = ()    # Auto-filled from compute kwargs?
    provides: str
    store: str = 'never'

    # Instance attributes that hold state
    run_name: str

    def key(self):
        return self.__class__.__name__, self.__version__

    def task_graph(self, run_name):
        depends_on = [PROVIDES[k] for k in self.depends_on]
        d = {self.key(): (self._compute,
                          run_name,
                          *[k.key() for k in depends_on])}
        # Update automatically overwrites, ensuring each extension
        # only appears once in the task graph
        # (even if it's requested multiple times)
        for k in self.depends_on:
            d.update(PROVIDES[k].task_graph(run_name))
        return d

    def _compute(self, run_name, *args):
        self.run_name = run_name
        return self.compute(*args)


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


# TODO: support for providing multiple things?
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
        d = np.zeros(len(peak_details), dtype=self.provides['peak_details'])
        d['width_full'] = peak_details['length'] * peak_details['dt']
        return d



# TODO: fill automatically with some register method
PROVIDES = dict(
    raw_records=RawRecords(),
    peak_details=PeakDetails(),
    peak_widths=PeakWidths(),
)
