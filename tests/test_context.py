import strax
from strax.testutils import (
    Records,
    Peaks,
    PeaksWoPerRunDefault,
    PeakClassification,
    run_id,
)
import tempfile
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as strategy
import typing as ty
import os
import unittest
import shutil
import uuid
import pytest


def _apply_function_to_data(function) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Inner core to test apply function to data.

    :param function: some callable function that takes thee positional arguments
    :return: records, records with function applied

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True), register=[Records]
        )

        # First load normal records
        records = st.get_array(run_id, "records")

        # Next update the context and apply
        st.set_context_config({"apply_data_function": function})
        changed_records = st.get_array(run_id, "records")
    return records, changed_records


def test_apply_pass_to_data():
    """What happens if we apply a function that does nothing (well, nothing hopefully)

    :return: None

    """

    def nothing(data, r, t):
        return data

    r, r_changed = _apply_function_to_data(nothing)
    assert np.all(r == r_changed)


def test_byte_strings_as_run_id():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True), register=[Records]
        )

        records_bytes = st.get_array(b"0", "records")
        records = st.get_array("0", "records")
        assert np.all(records_bytes == records)


@settings(deadline=None)
@given(strategy.integers(min_value=-10, max_value=10))
def test_apply_ch_shift_to_data(magic_shift: int):
    """Apply some magic shift number to the channel field and check the results.

    :param magic_shift: some number to check that we can shift the channel field with

    """

    def shift_channel(data, r, t):
        """Add a magic number to the channel field in the data."""
        res = data.copy()
        res["channel"] += magic_shift
        return res

    r, r_changed = _apply_function_to_data(shift_channel)
    assert len(r) == len(r_changed)
    assert np.all((r_changed["channel"] - (r["channel"] + magic_shift)) == 0)


def test_apply_drop_data():
    """What if we drop random portions of the data, do we get the right results?

    :return: None

    """

    class Drop:
        """Small class to keep track of the number of dropped rows."""

        kept = []

        def drop(self, data, r, t):
            """Drop a random portion of the data."""
            # I was too lazy to write a strategy to get the right number
            # of random drops based on the input records.
            keep = np.random.randint(0, 2, len(data)).astype(bool)

            # Keep in mind that we do this on a per chunk basis!
            self.kept += [keep]
            res = data.copy()[keep]
            return res

    # Init the bookkeeping class
    dropper = Drop()
    r, r_changed = _apply_function_to_data(dropper.drop)

    # The number of records should e
    assert np.all(r[np.concatenate(dropper.kept)] == r_changed)


def test_accumulate():
    """Test the st.accumulate function. Should add the results and accumulate per chunk. Lets add
    channels and verify the results are correct.

    :return: None

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        context = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records],
        )
        channels_from_array = np.sum(context.get_array(run_id, "records")["channel"])
        channels_accumulate = context.accumulate(run_id, "records", fields="channel")
        n_chunks = len(context.get_metadata(run_id, "records")["chunks"])
    channels = channels_accumulate["channel"]
    assert n_chunks == channels_accumulate["n_chunks"]
    assert channels_from_array == channels


def _get_context(temp_dir=tempfile.gettempdir()) -> strax.Context:
    """Get a context for the tests below."""
    context = strax.Context(
        storage=strax.DataDirectory(temp_dir, deep_scan=True),
        register=[Records, Peaks],
        use_per_run_defaults=True,
    )
    return context


def test_search_field():
    """Test search field in the context."""
    context = _get_context()
    context.search_field("data")


def test_show_config():
    """Test show_config in the context."""
    context = _get_context()
    df = context.show_config("peaks")
    assert len(df)


def test_data_itemsize():
    """Test data itemsize in the context."""
    context = _get_context()
    context.data_itemsize("peaks")


def test_data_info():
    """Test data info in the context."""
    context = _get_context()
    df = context.data_info("peaks")
    assert len(df)


def test_copy_to_frontend():
    """Write some data, add a new storage frontend and make sure that our copy to that frontend is
    successful."""
    # We need two directories for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        with tempfile.TemporaryDirectory() as temp_dir_2:
            context = _get_context(temp_dir)
            # Make some data
            context.get_array(run_id, "records")
            assert context.is_stored(run_id, "records")

            # Add the second frontend
            context.storage += [strax.DataDirectory(temp_dir_2)]
            # Test get_source_sf
            context.get_source_sf(run_id, ["records", "records"])
            context.copy_to_frontend(run_id, "records", target_compressor="lz4")

            # Make sure both frontends have the same data.
            assert os.listdir(temp_dir) == os.listdir(temp_dir)
            rec_folder = os.listdir(temp_dir)[0]
            list_temp_dir = sorted(os.listdir(os.path.join(temp_dir, rec_folder)))
            list_temp_dir_2 = sorted(os.listdir(os.path.join(temp_dir_2, rec_folder)))
            assert list_temp_dir == list_temp_dir_2

            # Clear the temp dir
            shutil.rmtree(temp_dir_2)

            # Now try again with rechunking
            context.copy_to_frontend(
                run_id,
                "records",
                target_compressor="lz4",
                rechunk_to_mb=400,
                rechunk=True,
            )


class TestContext(unittest.TestCase):
    """Test the per-run defaults options of a context."""

    def setUp(self):
        """Get a folder to write (Similar to tempfile.TemporaryDirectory) to get a temp folder in
        the temp dir.

        Cleanup after the tests.

        """
        temp_folder = uuid.uuid4().hex
        self.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        assert not os.path.exists(self.tempdir)

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_get_plugins_with_cache(self):
        st = self.get_context(False)
        st.register(Records)
        st.register(PeaksWoPerRunDefault)
        st.register(PeakClassification)

        not_cached_plugins = st._get_plugins(("peaks",), run_id)
        st._get_plugins(("peak_classification",), run_id)
        cached_plugins = st._get_plugins(("peaks",), run_id)
        assert (
            not_cached_plugins.keys() == cached_plugins.keys()
        ), f"_get_plugins returns different plugins if cached!"

    def test_multioutput_deregistration(self):
        """Test that a multi-output plugin is deregistered once one of its outputs is provided by
        another plugin."""

        class RecordsPlus(strax.Plugin):
            depends_on = tuple()
            data_kind = dict(records="records", plus="plus")
            dtype = dict(records=strax.record_dtype(), plus=strax.record_dtype())
            provides = ("records", "plus")

        st = self.get_context(False)
        st.register(RecordsPlus)
        st.register(Records)

        # Records will make the records, not RecordsPlus
        plugins = st._get_plugins(("records",), run_id)
        assert isinstance(plugins["records"], Records)

        # Plus is no longer available.
        with pytest.raises(KeyError):
            plugins = st.key_for("plus", run_id)

    def test_register_no_defaults(self, runs_default_allowed=False):
        """Test if we only register a plugin with no run-defaults."""
        st = self.get_context(runs_default_allowed)
        assert not self._has_per_run_default(Records)
        st.register(Records)
        st.select_runs()

    def test_register_no_defaults_but_allowed(self):
        self.test_register_no_defaults(runs_default_allowed=True)

    def test_register_with_defaults(self, runs_default_allowed=False):
        """Test if we register a plugin WITH run-defaults."""
        st = self.get_context(runs_default_allowed)
        assert not self._has_per_run_default(Records)
        assert self._has_per_run_default(Peaks)
        st.register(Records)
        if not runs_default_allowed:
            self.assertRaises(strax.InvalidConfiguration, st.register, Peaks)
        else:
            st.select_runs()

    def test_register_with_defaults_and_allowed(self):
        self.test_register_with_defaults(runs_default_allowed=True)

    def test_register_all_no_defaults(self, runs_default_allowed=False):
        """Test if we register a plugin with no run-defaults."""
        st = self.get_context(runs_default_allowed)
        assert any([self._has_per_run_default(Peaks), self._has_per_run_default(Records)])
        st.register(Records)

        if not runs_default_allowed:
            self.assertRaises(strax.InvalidConfiguration, st.register_all, strax.testutils)
        else:
            st.register_all(strax.testutils)

            st.select_runs()
        # Double check that peaks is registered in the context (otherwise the
        # test does not make sense)
        assert "peaks" in st._plugin_class_registry

    def test_register_all_no_defaults_and_allowed(self):
        self.test_register_all_no_defaults(runs_default_allowed=True)

    def test_deregister(self):
        """Tests if plugin cleaning is working:"""
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        st.deregister_plugins_with_missing_dependencies()
        assert all([p in st._plugin_class_registry for p in "peaks records".split()])
        st._plugin_class_registry.pop("records", None)
        st.deregister_plugins_with_missing_dependencies()
        assert st._plugin_class_registry.pop("peaks", None) is None

    def test_purge_configs(self):
        """Tests if config purging is working:"""
        st = self.get_context(True)
        st.set_config({"you_will_not_use_this_config": 42})
        st.purge_unused_configs()
        assert st.config.get("you_will_not_use_this_config", None) is None

    def get_context(self, use_defaults, **kwargs):
        """Get simple context where we have one mock run in the only storage frontend."""
        assert isinstance(use_defaults, bool)
        st = strax.Context(storage=self.get_mock_sf(), check_available=("records",), **kwargs)
        st.set_context_config({"use_per_run_defaults": use_defaults})
        return st

    def get_mock_sf(self):
        mock_rundb = [{"name": "0", strax.RUN_DEFAULTS_KEY: dict(base_area=43)}]
        sf = strax.DataDirectory(path=self.tempdir, deep_scan=True, provide_run_metadata=True)
        for d in mock_rundb:
            sf.write_run_metadata(d["name"], d)
        return sf

    def test_scan_runs__provided_dtypes__available_for_run(self):
        """Simple test with three plugins to test some basic context functions."""
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        st.register(PeakClassification)
        st.make(run_id, "records")

        # Test these three functions. They could have separate tests, but this
        # speeds things up a bit
        st.scan_runs()
        st.provided_dtypes()
        st.available_for_run(run_id)

    def test_bad_savewhen(self):
        st = self.get_context(True)

        class BadRecords(Records):
            save_when = "I don't know?!"

        st.register(BadRecords)
        with self.assertRaises(ValueError):
            st.get_save_when("records")
        with self.assertRaises(ValueError):
            st.get_single_plugin(run_id, "records")

    @staticmethod
    def _has_per_run_default(plugin) -> bool:
        """Does the plugin have at least one option that is a per-run default."""
        has_per_run_defaults = False
        for option in plugin.takes_config.values():
            has_per_run_defaults = option.default_by_run != strax.OMITTED
            if has_per_run_defaults:
                # Found one option
                break
        return has_per_run_defaults

    def test_get_source(self):
        """See if we get the correct answer for each of the plugins."""
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        st.register(self.get_dummy_peaks_dependency())
        st.set_context_config({"forbid_creation_of": ("peaks",)})
        for target in "records peaks cut_peaks".split():
            # Nothing is available and nothing should find a source
            assert not st.is_stored(run_id, target)
            assert st.get_source(run_id, target) is None

        # Now make a source "records"
        st.make(run_id, "records")
        assert st.get_source(run_id, "records") == {"records"}
        # since we cannot make peaks!
        assert st.get_source(run_id, "peaks", check_forbidden=True) is None
        assert st.get_source(run_id, "cut_peaks", check_forbidden=True) is None
        assert st.get_source(run_id, ("peaks", "cut_peaks"), check_forbidden=True) is None

        # We could ignore the error though
        assert st.get_source(run_id, "peaks", check_forbidden=False) == {"records"}
        assert st.get_source(run_id, "cut_peaks", check_forbidden=False) == {"records"}

        st.set_context_config({"forbid_creation_of": ()})
        st.make(run_id, "peaks")
        assert st.get_source(run_id, "records") == {"records"}
        assert st.get_source(run_id, ("records", "peaks")) == {"records", "peaks"}
        assert st.get_source(run_id, "peaks") == {"peaks"}
        assert st.get_source(run_id, "cut_peaks") == {"peaks"}

    def test_print_versions(self):
        """Test that we get that the "time" field from st.search_field."""
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        field = "time"
        field_matches, code_matches = st.search_field(field, return_matches=True)
        fields_found_for_dtype = [matched_field[0] for matched_field in field_matches[field]]
        self.assertTrue(
            all(p in fields_found_for_dtype for p in "records peaks".split()),
            f"{fields_found_for_dtype} is not correct, expected records or peaks",
        )
        self.assertTrue(
            all(p in code_matches[field] for p in "Records.compute Peaks.compute".split()),
            "code_matches[field] is not correct, expected Records.compute or Peaks.compute",
        )
        # Also test printing:
        self.assertIsNone(st.search_field(field, return_matches=False))

    def test_multi_run_loading_with_errors(self):
        st = self.get_context(True)
        st.register(Records)
        runs = [f"{i:06}" for i in range(10)]

        # make a copy and delete one random run
        make_runs = [r for r in runs]
        del make_runs[4]
        assert len(make_runs) < len(runs)

        for r in make_runs:
            st.make(r, "records")

        st.set_context_config(dict(forbid_creation_of="*"))
        with self.assertRaises(strax.DataNotAvailable):
            st.get_array(runs, "records", ignore_errors=False)
        records = st.get_array(runs, "records", ignore_errors=True)
        records_run_ids = np.unique(records["run_id"])
        assert all(r in records_run_ids for r in make_runs)
        assert set(runs) - set(make_runs) not in records_run_ids

    def test_auto_lineage(self):
        """Test that auto inferring a lineage from the plugin code works.

        Set auto-inferring version by setting __version__ = None for a plugin.

        """
        st = self.get_context(False)

        class DevelopRecords(Records):
            __version__ = None

        st.register(DevelopRecords)
        key = st.key_for(run_id, "records")
        plugin = st.get_single_plugin(run_id, "records")
        # Check that the version is auto-inferred
        assert plugin.version().startswith("auto_")

        # Check that loading and saving works as expected
        st.make(run_id, "records")
        assert st.is_stored(run_id, "records")
        st.set_context_config(dict(forbid_creation_of="*"))
        st.get_array(run_id, "records")

        # Plugin version (and therefore lineage) should not change when
        # making a new class. Let's try:
        class DevelopRecords(DevelopRecords):
            # New class, same properties (class name is encoded in
            # lineage!)
            pass

        st2 = st.new_context(register=DevelopRecords)
        assert key.lineage == st2.key_for(run_id, "records").lineage

        # When changing the class to have a different code, it should
        # get a new version and therefore a different lineage.
        class DevelopRecords(DevelopRecords):
            def compute(self, **kwargs):
                res = super().compute(**kwargs)
                return res

        st3 = st.new_context(register=DevelopRecords)
        assert key.lineage != st3.key_for(run_id, "records").lineage

        st.set_context_config(dict(allow_multiprocess=True, forbid_creation_of=tuple()))
        # Test that multithreaded processing works when the __version__ is None
        st.get_array([str(r) for r in range(10)], "records", max_workers=10)

    @staticmethod
    def get_dummy_peaks_dependency():
        class DummyDependsOnPeaks(strax.CutPlugin):
            depends_on = "peaks"
            provides = "cut_peaks"

        return DummyDependsOnPeaks

    def test_compare_metadata(self):
        st = self.get_context(True)
        st.register(Records)
        st.make(run_id, "records")
        old_metadata = st.get_metadata(run_id, "records")
        comparison_dict = st.compare_metadata(
            (run_id, "records"), old_metadata, return_results=True
        )
        comparison_dict["metadata2"].pop("strax_version")
        assert comparison_dict["metadata2"] == old_metadata, "metadata comparison failed"

    def test_get_data_kinds(self):
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        st.register(self.get_dummy_peaks_dependency())
        data_kind_collection, data_type_collection = st.get_data_kinds()
        # Assert the expected data kinds and their corresponding data types
        expected_data_kind_collection = {"records": ["records"], "peaks": ["peaks", "cut_peaks"]}
        assert data_kind_collection == expected_data_kind_collection
        # Assert the expected data types and their corresponding data kinds
        expected_data_type_collection = {
            "records": "records",
            "peaks": "peaks",
            "cut_peaks": "peaks",
        }
        assert data_type_collection == expected_data_type_collection

    def test_get_dependencies(self):
        st = self.get_context(True)
        st.register(Records)
        st.register(Peaks)
        st.register(self.get_dummy_peaks_dependency())
        assert "records" in st.get_dependencies("cut_peaks")
