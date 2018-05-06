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
    provides = 'raw_records'
    depends_on = tuple()
    dtype = strax.record_dtype()

    parallel = 'process'
    rechunk_on_save = False

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

    def _load_chunk(self, path, kind='central'):
        records = [strax.load_file(fn,
                                   compressor='blosc',
                                   dtype=strax.record_dtype())
                   for fn in glob.glob(f'{path}/reader_*')]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)
        if kind == 'central':
            return records
        result = strax.from_break(records,
                                  safe_break=int(1e3),  # TODO config?
                                  left=kind == 'post',
                                  tolerant=True)
        if self.config['erase']:
            shutil.rmtree(path)
        return result

    def compute(self, chunk_i):
        pre, current, post = self._chunk_paths(chunk_i)
        records = np.concatenate(
            ([self._load_chunk(pre, kind='pre')] if pre else [])
            + [self._load_chunk(current)]
            + ([self._load_chunk(post, kind='post')] if post else [])
        )

        strax.baseline(records)
        strax.integrate(records)

        timespan_sec = (records[-1]['time'] - records[0]['time']) / 1e9
        print(f'{chunk_i}: read {records.nbytes/1e6:.2f} MB '
              f'({len(records)} records, '
              f'{timespan_sec:.2f} sec) from readers')

        return records


@export
class Records(strax.Plugin):
    depends_on = ('raw_records',)
    data_kind = 'records'   # TODO: indicate cuts have been done?
    compressor = 'zstd'
    parallel = True
    rechunk_on_save = False
    dtype = strax.record_dtype()

    def compute(self, raw_records):
        r = strax.exclude_tails(raw_records, to_pe)
        hits = strax.find_hits(r)
        strax.cut_outside_hits(r, hits)
        return r


@export
@strax.takes_config(
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"))
class Peaks(strax.Plugin):
    depends_on = ('records',)
    data_kind = 'peaks'
    parallel = True
    rechunk_on_save = False
    dtype = strax.peak_dtype(n_channels=len(to_pe))

    def compute(self, records):
        r = records
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
    depends_on = ('peaks',)
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
@strax.takes_config(
    strax.Option('s1_max_width', default=150,
                 help="Maximum (IQR) width of S1s"),
    strax.Option('s1_min_n_channels', default=3,
                 help="Minimum number of PMTs that must contribute to a S1"),
    strax.Option('s2_min_area', default=100,
                 help="Minimum area (PE) for S2s"),
    strax.Option('s2_min_width', default=200,
                 help="Minimum width for S2s"))
class PeakClassification(strax.Plugin):
    __version__ = '0.0.1'
    depends_on = ('peak_basics',)
    dtype = [
        ('type', np.int8, 'Classification of the peak.')]
    parallel = True

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), dtype=self.dtype)

        is_s1 = p['n_channels'] > self.config['s1_min_n_channels']
        is_s1 &= p['range_50p_area'] < self.config['s1_max_width']
        r['type'][is_s1] = 1

        is_s2 = p['area'] > self.config['s2_min_area']
        is_s2 &= p['range_50p_area'] > self.config['s2_min_width']
        r['type'][is_s2] = 2

        return r


@strax.takes_config(
    strax.Option('min_area_fraction', default=0.5,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('nearby_window', default=int(1e7),
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'),
    strax.Option('ignore_below', default=10,
                 help='Peaks smaller than this are ignored completely. '
                      'This is necessary to have "safe breaks" in the data.')
)
class NCompeting(strax.Plugin):
    depends_on = ('peak_basics',)
    dtype = [
        ('n_competing', np.int32,
            'Number of nearby similarly-sized (or larger) peaks')]

    def rechunk_input(self, iters):
        return dict(peaks=strax.chunk_by_break(
            iters['peaks'],
            safe_break=self.config['nearby_window'],
            ignore_below=self.config['ignore_below']
        ))

    def compute(self, peaks):
        # TODO: allow dict of arrays output
        result = np.zeros(len(peaks), dtype=self.dtype)
        result['n_competing'] = find_n_competing(
            peaks,
            window=self.config['nearby_window'],
            fraction=self.config['min_area_fraction'],
            ignore_below=self.config['ignore_below'])
        return result


@numba.jit(nopython=True, nogil=True, cache=True)
def find_n_competing(peaks, window, fraction, ignore_below):
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
        results[i] = np.sum(a[left_i:right_i + 1] > max(ignore_below,
                                                        a[i] * fraction))

    return results - 1


def find_peak_groups(peaks, gap_threshold,
                     left_extension=0, right_extension=0,
                     max_duration=int(1e9)):
    # Mock up a "hits" array so we can just use the existing peakfinder
    # It doesn't work on raw peaks, since they might have different dts
    # TODO: is there no cleaner way?
    fake_hits = np.zeros(len(peaks), dtype=strax.hit_dtype)
    fake_hits['dt'] = 1
    fake_hits['time'] = peaks['time']
    # TODO: could this cause int nonsense?
    fake_hits['length'] = peaks['endtime'] - peaks['time']
    fake_peaks = strax.find_peaks(
        fake_hits, to_pe=np.zeros(1),
        gap_threshold=gap_threshold,
        left_extension=left_extension, right_extension=right_extension,
        min_hits=1, min_area=0,
        max_duration=max_duration)
    # TODO: cleanup input of meaningless fields?
    # (e.g. sum waveform)
    return fake_peaks


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
    strax.Option('max_event_duration', default=int(1e7),
                 help='Events longer than this are forcefully ended, '
                      'triggers in the truncated part are lost!'),
)
class Events(strax.Plugin):
    depends_on = ['peak_basics', 'n_competing']
    data_kind = 'events'
    dtype = [
        ('event_number', np.int64, 'Event number in this dataset'),
        ('time', np.int64, 'Event start time in ns since the unix epoch'),
        ('endtime', np.int64, 'Event end time in ns since the unix epoch')]
    parallel = False    # Since keeping state (events_seen)
    events_seen = 0

    def rechunk_input(self, iters):
        return dict(peaks=strax.chunk_by_break(
            iters['peaks'],
            safe_break=(self.config['left_extension']
                        + self.config['right_extension'] + 1),
            ignore_below=self.config['trigger_threshold']
        ))

    def compute(self, peaks):
        le = self.config['left_extension']
        re = self.config['right_extension']

        triggers = peaks[
            (peaks['area'] > self.config['trigger_threshold'])
            & (peaks['n_competing'] <= self.config['max_competing'])
            ]

        # Join nearby triggers
        peak_groups = find_peak_groups(
            triggers,
            gap_threshold=le + re + 1,
            left_extension=le,
            right_extension=re,
            max_duration=self.config['max_event_duration'])

        result = np.zeros(len(peak_groups), self.dtype)
        result['time'] = peak_groups['time']
        result['endtime'] = (peak_groups['time']
                             + peak_groups['length'] * peak_groups['dt'])
        result['event_number'] = (np.arange(len(result))
                                  + self.events_seen)

        # Filter out events without S1 + S2
        self.events_seen += len(result)

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


# LargestS2Area is just an example plugin
# The same info is provided in event_basics

class LargestS2Area(strax.LoopPlugin):
    """Find the largest S2 area in the event.
    """
    __version__ = '0.1'
    depends_on = ('events', 'peak_basics', 'peak_classification')

    dtype = [
        ('largest_s2_area', np.float32,
            'Area (PE) of largest S2 in event')]

    def compute_loop(self, event, peaks):

        s2s = peaks[peaks['type'] == 2]

        result = 0
        if len(s2s):
            result = s2s['area'].max()

        return dict(largest_s2_area=result)
