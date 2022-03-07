import shutil
from unittest import TestCase

import strax
from strax.testutils import Records, Peaks, PeakClassification, run_id


class ParrallelPeaks(Peaks):
    parallel = 'process'


class ParrallelPeakClassification(PeakClassification):
    parallel = 'process'


class ParralelEnds(strax.Plugin):
    """The most stupid plugin to make sure that we depend on _some_ of the output of ParrallelPeakClassification"""
    parallel = 'process'
    depends_on = 'peak_classification'
    dtype = strax.time_fields

    def compute(self, peaks):
        return {'time': peaks['time'], 'endtime': strax.endtime(peaks)}


class TestInline(TestCase):
    store_at = './.test_inline'

    def setUp(self) -> None:
        st = strax.context.Context(
            allow_multiprocess=True,
            allow_lazy=False,
            max_messages=4,
            timeout=60,
            config=dict(bonus_area=9),
        )
        st.storage = [strax.DataDirectory(self.store_at)]
        for p in [Records, ParrallelPeaks, ParrallelPeakClassification, ParralelEnds]:
            st.register(p)
        self.st = st
        assert not any(st.is_stored(run_id, t) for t in st._plugin_class_registry.keys())

    def tearDown(self) -> None:
        shutil.rmtree(self.store_at)

    def test_inline(self):
        st = self.st
        targets = list(st._plugin_class_registry.keys())
        st.make(run_id,
                list(targets),
                allow_multiple=True,
                max_workers=2,
                config=dict(bonus_area=10),
                )
        for target in targets:
            if st.get_save_when(target) == strax.SaveWhen.ALWAYS:
                assert st.is_stored(run_id, target)
