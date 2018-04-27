from .common import to_pe

import numpy as np

import strax
export, __all__ = strax.exporter()


@export
class ReducedRecords(strax.Plugin):
    data_kind = 'records'
    compressor = 'zstd'
    parallel = True
    rechunk = False
    dtype = strax.record_dtype()

    def compute(self, records):
        r = records
        strax.integrate(r)
        r = strax.exclude_tails(r, to_pe)
        return r


@export
class Peaks(strax.Plugin):
    data_kind = 'peaks'
    parallel = True
    rechunk = False
    dtype = strax.peak_dtype(n_channels=len(to_pe))

    def compute(self, reduced_records):
        r = reduced_records

        hits = strax.find_hits(r)
        strax.cut_outside_hits(r, hits)

        peaks = strax.find_peaks(hits, to_pe,
                                 result_dtype=self.dtype)
        strax.sum_waveform(peaks, r, to_pe)

        peaks = strax.split_peaks(peaks, r, to_pe)

        strax.compute_widths(peaks)
        return peaks


@export
class PeakBasics(strax.Plugin):
    parallel = True
    dtype = [
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Start time of the peak (ns since unix epoch)',
            'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
            'endtime'), np.int64),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Fraction of area seen by the top array',
            'area_fraction_top'), np.float32),
    ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        r['area'] = p['area']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['time'] = p['time']
        r['endtime'] = p['time'] + p['dt'] * p['length']

        # TODO: get n_top_pmts from some config...
        area_top = (p['area_per_channel'][:, :127]
                    * to_pe[:127].reshape(1, -1)).sum(axis=1)
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

        is_s1 = p['area'] > 100
        is_s1 &= p['range_50p_area'] < 150
        r['type'][is_s1] = 1

        is_s2 = p['area'] > 1e4
        is_s2 &= p['range_50p_area'] > 200
        r['type'][is_s2] = 2

        return r


@export
class Events(strax.Plugin):
    data_kind = 'events'
    save_preference = strax.SaveWhen.ALWAYS
    dtype = [
        (('Event number in this dataset',
          'event_number'), np.int64),
        (('Event start time in ns since the unix epoch',
          'time'), np.int64),
        (('Event end time in ns since the unix epoch',
          'endtime'), np.int64),
    ]
    parallel = False

    # Uh oh, state... must force sequential when we start doing multiprocessing
    events_seen = 0

    def compute(self, peak_basics):
        left_ext = int(1e6)
        right_ext = int(1e6)
        large_peaks = peak_basics[peak_basics['area'] > 1e5]

        # TODO: this can be done much faster
        event_ranges = []
        split_indices = np.where(np.diff(large_peaks['time'])
                                 > left_ext + right_ext)[0] + 1
        for ps in np.split(large_peaks, split_indices):
            start = ps[0]['time'] - left_ext
            stop = ps[-1]['time'] + right_ext
            event_ranges.append((start, stop))
        event_ranges = np.array(event_ranges)
        self.events_seen += len(event_ranges)

        result = np.zeros(len(event_ranges), self.dtype)
        result['time'], result['endtime'] = event_ranges.T
        result['event_number'] = (np.arange(len(event_ranges))
                                  + self.events_seen)
        return result


@export
class EventBasics(strax.LoopPlugin):
    depends_on = ('events', 'peak_basics', 'peak_classification')
    dtype = [(('Number of peaks in the event',
               'n_peaks'), np.int32),

             (('Main S1 peak index',
               's1_index'), np.int32),
             (('Main S1 area (PE)',
               's2_area'), np.int32),
             (('Main S1 area fraction top',
               's1_area_fraction_top'), np.float32),
             (('Main S1 width (ns, 50% area)',
               's1_range_50p_area'), np.float32),

             (('Main S2 peak index',
               's2_index'), np.int32),
             (('Main S2 area (PE)',
               's1_area'), np.int32),
             (('Main S2 area fraction top',
               's2_area_fraction_top'), np.float32),
             (('Main S2 width (ns, 50% area)',
               's2_range_50p_area'), np.float32),

             (('Drift time between main S1 and S2 in ns',
               'drift_time'), np.int64),
             ]

    def compute_loop(self, event, peaks):
        result = dict(n_peaks=len(peaks))
        if not len(peaks):
            return result

        main_s = dict()
        for s_i in [1, 2]:
            ss = peaks[peaks['type'] == s_i]
            if not len(ss):
                continue
            main_i = result[f's{s_i}_index'] = np.argmax(ss['area'])
            s = main_s[s_i] = ss[main_i]
            for prop in 'area area_fraction_top range_50p_area'.split():
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
