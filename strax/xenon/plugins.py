import glob
import os
import time
import shutil

import numpy as np
import numba

import strax
from .common import to_pe
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('input_dir', type=str, track=False,
                 help="Directory where readers put data"),
    strax.Option('erase', default=False, track=False,
                 help="Delete reader data after processing"))
class DAQReader(strax.Plugin):
    provides = 'records'
    dtype = strax.record_dtype()

    parallel = 'process'
    rechunk = False
    can_remake = False

    def _path(self, chunk_i):
        return self.config["input_dir"] + f'/{chunk_i:06d}'

    def _chunk_paths(self, chunk_i):
        """Return paths to previous, current and next chunk
        If any of them does not exist, their path is replaced by False.
        """
        p = self._path(chunk_i)
        return tuple([
            q if os.path.exists(q) else False
            for q in [p + '_pre', p, p + '_post']])

    def check_next_ready_or_done(self, chunk_i):
        while True:
            ended = os.path.exists(self.config["input_dir"] + f'/THE_END')
            pre, current, post = self._chunk_paths(chunk_i)
            next_ahead = os.path.exists(self._path(chunk_i + 1))
            if (current and (
                    (pre and post
                     or chunk_i == 0 and post
                     or ended and (pre and not next_ahead)))):
                return True
            if ended and not current:
                return False
            print(f"Waiting for chunk {chunk_i}, sleeping")
            time.sleep(2)

    @staticmethod
    def _load_chunk(path, kind='central'):
        records = [strax.load_file(fn,
                                   compressor='zstd',
                                   dtype=strax.record_dtype())
                   for fn in glob.glob(f'{path}/reader_*')]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)
        if kind == 'central':
            return records
        return strax.from_break(records, left=kind == 'post')

    def compute(self, chunk_i):
        pre, current, post = self._chunk_paths(chunk_i)
        records = np.concatenate(
            ([self._load_chunk(pre, kind='pre')] if pre else [])
            + [self._load_chunk(current)]
            + ([self._load_chunk(post, kind='post')] if post else [])
        )

        if self.config['erase']:
            for x in pre, current, post:
                shutil.rmtree(x)

        strax.baseline(records)
        strax.integrate(records)

        return records


@export
class ReducedRecords(strax.Plugin):
    data_kind = 'records'   # TODO: indicate cuts have been done?
    compressor = 'zstd'
    parallel = True
    rechunk = False
    dtype = strax.record_dtype()

    def compute(self, records):
        r = strax.exclude_tails(records, to_pe)
        hits = strax.find_hits(r)
        strax.cut_outside_hits(r, hits)
        return r


@export
@strax.takes_config(
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"))
class Peaks(strax.Plugin):
    data_kind = 'peaks'
    parallel = True
    rechunk = False
    dtype = strax.peak_dtype(n_channels=len(to_pe))

    def compute(self, reduced_records):
        r = reduced_records
        hits = strax.find_hits(r)       # TODO: Duplicate work
        hits = strax.sort_by_time(hits)

        peaks = strax.find_peaks(hits, to_pe,
                                 result_dtype=self.dtype)
        strax.sum_waveform(peaks, r, to_pe)

        peaks = strax.split_peaks(peaks, r, to_pe)

        strax.compute_widths(peaks)

        if self.config['diagnose_sorting']:
            assert np.diff(r['time']).min() >= 0, "Records not sorted"
            assert np.diff(hits['time']).min() >= 0, "Hits not sorted"
            assert np.all(peaks['time'][1:]
                          >= strax.endtime(peaks)[:-1]), "Peaks not disjoint"

        return peaks


@export
class PeakBasics(strax.Plugin):
    parallel = True
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Fraction of area seen by the top array',
            'area_fraction_top'), np.float32),

        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
    ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        for q in 'time length dt area'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)

        # TODO: get n_top_pmts from config...
        area_top = (p['area_per_channel'][:, :127]
                    * to_pe[:127].reshape(1, -1)).sum(axis=1)
        # Negative-area peaks get 0 AFT - TODO why not NaN?
        m = p['area'] > 0
        r['area_fraction_top'][m] = area_top[m]/p['area'][m]
        return r


@export
class PeakClassification(strax.Plugin):
    parallel = True
    dtype = [
        (('Classification of the peak.',
          'type'), np.int8)
    ]

    def compute(self, peak_basics):
        p = peak_basics
        r = np.zeros(len(p), dtype=self.dtype)

        is_s1 = p['area'] > 2
        is_s1 &= p['range_50p_area'] < 150
        r['type'][is_s1] = 1

        is_s2 = p['area'] > 100
        is_s2 &= p['range_50p_area'] > 200
        r['type'][is_s2] = 2

        return r


@strax.takes_config(
    strax.Option('max_competitive', default=7,
                 help='Peaks with >= this number of similarly-sized or higher '
                      'peaks do not trigger events'),
    strax.Option('min_area_fraction', default=0.5,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('max_time_diff', default=int(1e7),
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'))
class NCompeting(strax.Plugin):
    # TODO: Need overlap handling!
    parallel = True
    dtype = [
        (('Number of nearby similarly-sized (or larger) peaks',
          'n_competing'), np.int32),
    ]

    def compute(self, peak_basics):
        # TODO: allow dict of arrays output
        result = np.zeros(len(peak_basics), dtype=self.dtype)
        result['n_competing'] = find_n_competing(
            peak_basics,
            window=self.config['max_time_diff'],
            fraction=self.config['min_area_fraction'])
        return result


@numba.jit(nopython=True, nogil=True, cache=True)
def find_n_competing(peaks, window, fraction):
    n = len(peaks)
    t = peaks['time']
    a = peaks['area']
    results = np.zeros(n, dtype=np.int32)

    left_i = 0
    right_i = 0
    for i, peak in enumerate(peaks):
        while t[left_i] + window < t[i] and left_i < n - 1:
            left_i += 1
        while t[right_i] - window < t[i] and right_i < n - 1:
            right_i += 1
        results[i] = np.sum(a[left_i:right_i + 1] > a[i] * fraction) - 1

    return results


@strax.takes_config(
    strax.Option('trigger_threshold', default=100,
                 help='Peaks with area (PE) below this do NOT cause events'),
    strax.Option('max_competing', default=7,
                 help='Peaks with  this number of similarly-sized or higher '
                      'peaks do NOT cause events'),
    strax.Option('left_extension', default=int(1e6),
                 help='Extend events this many ns to the left from each '
                      'triggering peak'),
    strax.Option('right_extension', default=int(1e6),
                 help='Extend events this many ns to the right from each '
                      'triggering peak'),
    strax.Option('max_event_length', default=int(1e7),
                 help='Events longer than this are forcefully ended, '
                      'triggers in the truncated part are lost!')
)
class Events(strax.Plugin):
    data_kind = 'events'
    dtype = [
        (('Event number in this dataset',
          'event_number'), np.int64),
        (('Event start time in ns since the unix epoch',
          'time'), np.int64),
        (('Event end time in ns since the unix epoch',
          'endtime'), np.int64),
    ]
    parallel = False  # Since keeping state (events_seen)
    events_seen = 0

    # TODO: automerge deps
    def compute(self, peak_basics, n_competing):
        le = self.config['left_extension']
        re = self.config['right_extension']

        triggers = peak_basics[
            (peak_basics['area'] > self.config['trigger_threshold'])
            & (n_competing['n_competing'] <= self.config['max_competing'])
            ]

        # Join nearby triggers - yet another max-gap clustering
        # Mock up a "hits" array so we can just use the existing peakfinder
        # TODO: is there no cleaner way?
        fake_hits = np.zeros(len(triggers), dtype=strax.hit_dtype)
        fake_hits['dt'] = 1
        fake_hits['time'] = triggers['time']
        # TODO: could this cause int nonsense?
        fake_hits['length'] = triggers['endtime'] - triggers['time']
        fake_peaks = strax.find_peaks(
            fake_hits, to_pe=np.zeros(1),
            gap_threshold=le + re + 1,
            left_extension=le, right_extension=re,
            min_hits=1, min_area=0,
            max_duration=self.config['max_event_length'])

        result = np.zeros(len(fake_peaks), self.dtype)
        result['time'] = fake_peaks['time']
        result['endtime'] = (fake_peaks['time']
                             + fake_peaks['length'] * fake_peaks['dt'])
        result['event_number'] = (np.arange(len(result))
                                  + self.events_seen)

        # Filter out events without S1 + S2
        self.events_seen += len(result)

        # TODO dirty hack!!
        # This will mangle events to ensure they are nonoverlapping
        # Needed for online processing until we have overlap handling...
        # Alternative is to put n_per_iter = float('inf')
        result['time'] = np.clip(result['time'],
                                 peak_basics[0]['time'], None)
        result['endtime'] = np.clip(result['endtime'],
                                    None, peak_basics[-1]['endtime'])
        return result
        # TODO: someday investigate why loopplugin doesn't give
        # anything if events do not contain peaks..


@export
class EventBasics(strax.LoopPlugin):
    depends_on = ('events',
                  'peak_basics', 'peak_classification', 'n_competing')

    def infer_dtype(self):
        dtype = [(('Number of peaks in the event',
                   'n_peaks'), np.int32),
                 (('Drift time between main S1 and S2 in ns',
                   'drift_time'), np.int64)]
        for i in [1, 2]:
            dtype += [((f'Main S{i} peak index',
                        f's{i}_index'), np.int32),
                      ((f'Main S{i} area (PE)',
                        f's{i}_area'), np.float32),
                      ((f'Main S{i} area fraction top',
                        f's{i}_area_fraction_top'), np.float32),
                      ((f'Main S{i} width (ns, 50% area)',
                        f's{i}_range_50p_area'), np.float32),
                      ((f'Main S{i} number of competing peaks',
                        f's{i}_n_competing'), np.int32)]
        return dtype

    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks))
        if not len(peaks):
            return result

        main_s = dict()
        for s_i in [1, 2]:
            s_mask = peaks['type'] == s_i
            ss = peaks[s_mask]
            s_indices = np.arange(len(peaks))[s_mask]
            if not len(ss):
                result[f's{s_i}_index'] = -1
                continue
            main_i = np.argmax(ss['area'])
            result[f's{s_i}_index'] = s_indices[main_i]
            s = main_s[s_i] = ss[main_i]
            for prop in ['area', 'area_fraction_top',
                         'range_50p_area', 'n_competing']:
                result[f's{s_i}_{prop}'] = s[prop]

        if len(main_s) == 2:
            result['drift_time'] = main_s[2]['time'] - main_s[1]['time']

        return result


@export
class LargestPeakArea(strax.LoopPlugin):
    depends_on = ('events', 'peak_basics')
    dtype = [(('Area of largest peak in event (PE)',
               'largest_area'), np.float32)]

    def compute_loop(self, event, peaks):
        x = 0
        if len(peaks):
            x = peaks['area'].max()

        return dict(largest_area=x)
