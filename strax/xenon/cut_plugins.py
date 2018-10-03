import numpy as np

import strax
export, __all__ = strax.exporter()


class FiducialCylinder1T(strax.Plugin):
    """Implementation of fiducial volume cylinder 1T,
    ported from lax.sciencerun0.py"""
    depends_on = ('event_positions',)
    provides = 'fiducial_cylinder_1t'
    dtype = [('cut_fiducial_cylinder', np.bool, 'One tonne fiducial cylinder')]

    def compute(self, events):
        arr = np.all([(-92.9 < events['z']), (-9 > events['z']),
                      (36.94 > np.sqrt(events['x']**2 + events['y']**2))],
                     axis=0)
        return dict(cut_fiducial_cylinder=arr)


class S1MaxPMT(strax.LoopPlugin):
    """Removes events where the largest hit in S1 is too large
    port from lax.sciencerun0.py"""
    depends_on = ('events', 'event_basics', 'peak_basics')
    dtype = [('cut_s1_max_pmt', np.bool, 'S1 max PMT cut')]

    def compute_loop(self, event, peaks):
        ret = dict(cut_s1_max_pmt=True)
        if not len(peaks) or np.isnan(event['s1_index']):
            return ret

        peak = peaks[event['s1_index']]
        max_channel = peak['max_pmt_area']
        ret['cut_s1_max_pmt'] = (max_channel < 0.052 * event['s1_area'] + 4.15)
        return ret


class S1LowEnergyRange(strax.Plugin):
    """Pass only events with cs1<200"""
    depends_on = ('events', 'corrected_areas')
    dtype = [('cut_s1_low_energy_range', np.bool, "Event under 200pe")]

    def compute(self, events):
        ret = np.all([events['cs1'] < 200], axis=0)
        return dict(cut_s1_low_energy_range=ret)


class SR1Cuts(strax.MergeOnlyPlugin):
    depends_on = ['fiducial_cylinder_1t', 's1_max_pmt', 's1_low_energy_range']
    save_when = strax.SaveWhen.ALWAYS


class FiducialEvents(strax.Plugin):
    depends_on = ['event_info', 'fiducial_cylinder_1t']
    data_kind = 'fiducial_events'

    def infer_dtype(self):
        return strax.merged_dtype([self.deps[d].dtype
                                   for d in self.depends_on])

    def compute(self, events):
        return events[events['cut_fiducial_cylinder']]
