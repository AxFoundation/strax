import shutil
from unittest import TestCase
import immutabledict
import strax
from strax.testutils import Records, Peaks, PeakClassification, run_id


class ParallelPeaks(Peaks):
    parallel = "process"


class ParallelPeakClassification(PeakClassification):
    parallel = "process"
    save_when = {k: strax.SaveWhen.EXPLICIT for k in PeakClassification.provides}
    save_when["lone_hits"] = strax.SaveWhen.ALWAYS
    save_when = immutabledict.immutabledict(save_when)


class ParallelEnds(strax.Plugin):
    """The most stupid plugin to make sure that we depend on _some_ of the output of
    ParallelPeakClassification."""

    parallel = "process"
    provides = "parallel_ends"
    depends_on = "peak_classification"
    dtype = strax.time_fields

    def compute(self, peaks):
        return {"time": peaks["time"], "endtime": strax.endtime(peaks)}


class TestInline(TestCase):
    store_at = "./.test_inline"

    def setUp(self) -> None:
        st = strax.context.Context(
            allow_multiprocess=True,
            allow_lazy=False,
            max_messages=4,
            timeout=60,
            config=dict(bonus_area=9),
        )
        st.storage = [strax.DataDirectory(self.store_at)]
        for p in [Records, ParallelPeaks, ParallelPeakClassification, ParallelEnds]:
            st.register(p)
        self.st = st
        assert not any(st.is_stored(run_id, t) for t in st._plugin_class_registry.keys())

    def tearDown(self) -> None:
        shutil.rmtree(self.store_at)

    def test_inline(self, **make_kwargs):
        st = self.st
        targets = ("records", "parallel_ends")
        st.make(
            run_id,
            targets,
            allow_multiple=True,
            **make_kwargs,
        )
        for target in list(st._plugin_class_registry.keys()):
            should_be_stored = st.get_save_when(target) == strax.SaveWhen.ALWAYS
            if target in targets and not should_be_stored:
                # redundant check but just in case someone ever changes
                # this test the records test plugin
                should_be_stored = st.get_save_when(target) == strax.SaveWhen.TARGET
            assert st.is_stored(run_id, target) == should_be_stored

    def test_inline_with_multi_processing(self, **make_kwargs):
        self.test_inline(max_workers=2, **make_kwargs)

    def test_inline_with_temp_config(self, **make_kwargs):
        self.test_inline_with_multi_processing(config=dict(secret_time_offset=10), **make_kwargs)

    def test_inline_bare(self, n_chunks=3):
        """Get the plugin from a bare processor and run in this thread."""
        st = self.st
        st.set_config(dict(n_chunks=n_chunks))
        targets = list(st._plugin_class_registry.keys())
        components = st.get_components(run_id, targets=targets)
        parallel_components = strax.ParallelSourcePlugin.inline_plugins(
            components, start_from="records", log=st.log
        )
        parallel_plugin = parallel_components.plugins["parallel_ends"]
        for chunk_i in range(n_chunks):
            assert len(parallel_plugin.do_compute(chunk_i=chunk_i))
