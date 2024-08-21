import os
import re
import json
from bson import json_util
import tempfile
import shutil
import pytz
import datetime

import unittest
import numpy as np
import pandas as pd

import strax
from strax import Plugin, ExhaustPlugin
from strax.testutils import Records, Peaks, PeakClassification


class TestSuperRuns(unittest.TestCase):
    def setUp(self, superrun_name="_superrun_test"):
        self.offset_between_subruns = 10
        self.superrun_name = superrun_name
        self.subrun_modes = ["mode_a", "mode_b"]
        self.subrun_source = "test"
        # Temp directory for storing record data for the tests.
        # Will be removed during TearDown.
        self.tempdir = tempfile.mkdtemp()
        self.tempdir2 = tempfile.mkdtemp()  # Required to test writing superruns
        # with two storage frontends
        self.context = strax.Context(
            storage=[
                strax.DataDirectory(
                    self.tempdir, provide_run_metadata=True, readonly=False, deep_scan=True
                )
            ],
            register=[
                Records,
                RecordsExtension,
                Peaks,
                PeakClassification,
                PeaksExtension,
                PeaksExtensionCopy,
            ]
            + [Ranges, Sum],
            config={
                "bonus_area": 42,
                "n_chunks": 1,
            },
            store_run_fields=(
                "name",
                "number",
                "start",
                "end",
                "livetime",
                "mode",
                "source",
            ),
        )
        self.context.set_context_config({"write_superruns": True, "use_per_run_defaults": False})

        logger = self.context.log
        logger.addFilter(
            lambda s: not re.match(".*Could not estimate run start and end time.*", s.getMessage())
        )
        self.context._plugin_class_registry["records"].chunk_target_size_mb = 1
        self._create_subruns()
        self.context.define_run(self.superrun_name, data=self.subrun_ids)

    def test_superrun_access(self):
        """Tests if storage fornt-ends which does not provide superruns raise correct exception."""
        self.context.storage[0].provide_superruns = False
        with self.assertRaises(strax.DataNotAvailable):
            self.context.storage[0].find(self.context.key_for(self.superrun_name, "peaks"))

    def test_run_meta_data(self):
        """Check if superrun has the correct run start/end and livetime and subruns are sroted by
        start times."""
        superrun_meta = self.context.run_metadata(self.superrun_name)
        subrun_meta = [self.context.run_metadata(r) for r in self.subrun_ids]

        assert superrun_meta["start"] == subrun_meta[0]["start"]
        assert superrun_meta["end"] == subrun_meta[-1]["end"]
        livetime = 0
        for meta in subrun_meta:
            time_delta = meta["end"] - meta["start"]
            livetime += time_delta.total_seconds()
        assert superrun_meta["livetime"] == livetime

        prev_start = datetime.datetime.min.replace(tzinfo=pytz.utc)
        for subrun_id in superrun_meta["sub_run_spec"]:
            start = self.context.run_metadata(subrun_id)["start"].replace(tzinfo=pytz.utc)
            assert start > prev_start, "Subruns should be sorted by run starts"
            prev_start = start

    def test_loaders_and_savers(self):
        """Tests if loaders and savers are correctly set for superruns.

        The context will dig the dependency tree till the plugin which does not allow_superrun, Then
        the subruns of data_type of that plugin will be collected and combined. The superrun is
        processed based on the combined subruns.

        """
        self.context.make(self.subrun_ids, "peaks")
        # Only if peak_classification depends on only peaks the test works
        assert len(self.context._plugin_class_registry["peak_classification"].depends_on) == 1
        assert all(self.context.is_stored(subrun_id, "peaks") for subrun_id in self.subrun_ids)
        # Although peaks are saved, processing
        # peak_classification's records will still starts from records
        components = self.context.get_components(self.superrun_name, "peak_classification")
        # Because records is not allow_superrun
        assert "records" in components.loaders
        # Because though we call for peak_classification,
        # peaks already allow_superrun
        assert "peaks" not in components.loaders
        # peaks and lone_hits should all be saved
        assert "peaks" in components.savers
        assert "lone_hits" in components.savers
        # of course peak_classification should be saved
        assert "peak_classification" in components.savers

        # When we make superrun, subruns of the targeted data_type should
        # be first made individually and combined.
        components = self.context.get_components(
            self.superrun_name, "peak_classification", _combining_subruns=True
        )
        assert len(components.loaders) == 1
        assert "peak_classification" in components.loaders

        with self.assertRaises(ValueError):
            self.context.get_components(
                self.superrun_name, ("peaks", "peak_classification"), _combining_subruns=True
            )

    def test_create_and_load_superruns(self):
        """Creates "new" superrun data from already existing data.

        Loads and compare data afterwards. Also tests "add_run_id_field" option.

        """

        subrun_data = self.context.get_array(
            self.subrun_ids, "peaks", progress_bar=False, add_run_id_field=False
        )
        self.context.make(self.superrun_name, "peaks")
        superrun_data = self.context.get_array(self.superrun_name, "peaks")

        assert self.context.is_stored(self.superrun_name, "peaks")
        assert np.all(subrun_data == superrun_data)

        # Load meta data and check if rechunking worked:
        chunks = self.context.get_meta(self.superrun_name, "peaks")["chunks"]
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["run_id"] == self.superrun_name
        assert chunk["first_time"] == subrun_data["time"].min()
        assert chunk["last_endtime"] == np.max(strax.endtime(subrun_data))

        # Check if subruns and superrun have the same time stamps
        self.context.set_context_config({"write_superruns": False})
        subrun_data = self.context.get_array(self.subrun_ids, "peaks", progress_bar=False)
        superrun_data = self.context.get_array(self.superrun_name, "peaks")
        assert np.all(subrun_data["time"] == superrun_data["time"])

    def test_select_runs_with_superruns(self):
        """Test if select_runs works correctly with superruns."""
        df = self.context.select_runs()
        mask_superrun = df["name"] == self.superrun_name
        assert pd.api.types.is_string_dtype(df["mode"])
        modes = df.loc[mask_superrun, "mode"].values[0]
        modes = modes.split(",")
        assert set(modes) == set(self.subrun_modes)

        assert pd.api.types.is_string_dtype(df["tags"])
        assert pd.api.types.is_string_dtype(df["source"])
        assert df.loc[mask_superrun, "source"].values[0] == "test"
        assert pd.api.types.is_timedelta64_dtype(df["livetime"])

        self.context.make(self.superrun_name, "peaks")
        df = self.context.select_runs(available=("peaks",))
        assert self.superrun_name in df["name"].values

    def test_superrun_chunk_properties(self):
        """Check properties of superrun's chunk information."""
        self.context.make(self.superrun_name, "peaks")

        # Load subrun and see if propeties work:
        for chunk in self.context.get_iter(self.subrun_ids[0], "peaks"):
            assert not chunk.is_superrun
            assert not chunk.first_subrun
            assert not chunk.last_subrun

        # Now for a superrun
        for chunk in self.context.get_iter(self.superrun_name, "peaks"):
            assert chunk.is_superrun
            subruns = chunk.subruns
            run_ids = list(subruns.keys())
            first_subrun = chunk.first_subrun
            last_subrun = chunk.last_subrun
            _is_ordered = first_subrun["run_id"] == run_ids[0]
            _is_ordered &= last_subrun["run_id"] == run_ids[-1]
            assert _is_ordered, "Subruns dictionary appears to be not ordered correctly!"
            for ind, _subruns in zip([0, -1], [first_subrun, last_subrun]):
                assert _subruns["start"] == subruns[run_ids[ind]]["start"]
                assert _subruns["end"] == subruns[run_ids[ind]]["end"]

    def test_superrun_definition(self):
        """Test if superrun definition works correctly.

        After redefining the superrun, the DataKey of superrun will be different.

        """
        self.context.get_array(self.superrun_name, "peaks")
        assert self.context.is_stored(self.superrun_name, "peaks")
        # After redefining the superrun with only different subruns,
        # the superrun should not be stored
        self.context.define_run(self.superrun_name, data=self.subrun_ids[:-1])
        assert not self.context.is_stored(self.superrun_name, "peaks")
        self.context.define_run(self.superrun_name, data=self.subrun_ids)

    def test_superrun_chunk_and_meta(self):
        """Superrun chunks and meta data should contain information about its constituent
        subruns."""
        self.context.make(self.superrun_name, "peaks")
        meta = self.context.get_meta(self.superrun_name, "peaks")

        n_chunks = 0
        superrun_chunk = None
        for chunk in self.context.get_iter(self.superrun_name, "peaks"):
            superrun_chunk = chunk
            n_chunks += 1

        assert len(meta["chunks"]) == n_chunks == 1
        assert superrun_chunk.subruns is not None
        assert meta["chunks"][0]["subruns"] == superrun_chunk.subruns

        for subrun_id, start_and_end in superrun_chunk.subruns.items():
            rr = self.context.get_array(subrun_id, "peaks")
            # Tests below only true for peaks as we have not rechunked yet.
            # After rechunking in general data start can be different from chunk start
            mes = f"Start time did not match for subrun: {subrun_id}"
            assert rr["time"].min() == start_and_end["start"], mes
            mes = f"End time did not match for subrun: {subrun_id}"
            assert np.max(strax.endtime(rr)) == start_and_end["end"], mes

    def test_rechnunking_and_loading(self):
        """Tests rechunking and loading of superruns with multiple chunks.

        The test is required since it was possible to run into race conditions with
        chunk.continuity_check in context.get_iter.

        """

        self.context.set_config({"recs_per_chunk": 500})  # Make chunks > 1 MB

        rr = self.context.get_array(self.subrun_ids, "peaks")
        endtime = np.max(strax.endtime(rr))

        # Make two additional chunks which are large comapred to the first
        # three chunks
        next_subrun_id = int(self.subrun_ids[-1]) + 1
        for run_id in range(next_subrun_id, next_subrun_id + 2):
            self.context.set_config(
                {"secret_time_offset": int(endtime + self.offset_between_subruns)}
            )
            rr = self.context.get_array(str(run_id), "peaks")
            self._write_run_doc(
                run_id,
                self.now + datetime.timedelta(0, int(rr["time"].min())),
                self.now + datetime.timedelta(0, int(np.max(strax.endtime(rr)))),
            )
            endtime = np.max(strax.endtime(rr))
            self.subrun_ids.append(str(run_id))

        superrun_test_rechunking = "_superrun_test_rechunking"
        self.context.define_run(superrun_test_rechunking, self.subrun_ids)
        self.context.make(superrun_test_rechunking, "peaks")
        assert self.context.is_stored(superrun_test_rechunking, "peaks")

        rr_superrun = self.context.get_array(superrun_test_rechunking, "peaks")
        rr_subruns = self.context.get_array(self.subrun_ids, "peaks")

        assert np.all(rr_superrun["time"] == rr_subruns["time"])

    def test_superrun_triggers_subrun_processing(self):
        """Tests if superrun processing can trigger subrun processing.

        Which it should.

        """
        self.context._plugin_class_registry["peaks"].allow_superrun = False
        assert not self.context.is_stored(self.superrun_name, "peaks_extension_copy")
        assert not self.context.is_stored(self.subrun_ids[0], "peaks_extension")

        self.context.make(self.superrun_name, "peaks_extension_copy", save="peaks_extension_copy")
        assert self.context.is_stored(self.superrun_name, "peaks_extension_copy")
        # Only the highest level for save_when.EXPLICIT plugins should be stored.
        assert not self.context.is_stored(self.subrun_ids[0], "peaks_extension")
        assert self.context.is_stored(self.subrun_ids[0], "peaks")

    def test_storing_with_second_sf(self):
        """Tests if only superrun is written to new sf if subruns already exist in different sf."""
        self.context.storage[0].readonly = True
        self.context.storage.append(strax.DataDirectory(self.tempdir2, provide_run_metadata=True))
        self.context.make(self.superrun_name, "peaks")
        superrun_sf = self.context.storage.pop(1)
        # Check if first sf contains superrun, it should not:
        assert not self.context.is_stored(self.superrun_name, "peaks")

        # Now check second sf for which only the superrun should be stored
        self.context.storage = [superrun_sf]
        self._create_subruns()
        self.context.define_run(self.superrun_name, data=self.subrun_ids)
        assert self.context.is_stored(self.superrun_name, "peaks")

    def test_only_combining_superruns(self):
        """Test loading superruns when only combining subruns.

        The test also shows the difference between the two.

        """
        self.context.check_superrun()
        sum_super = self.context.get_array(self.superrun_name, "sum")
        _sum_super = self.context.get_array(self.superrun_name, "sum", _combining_subruns=True)

        # superruns will still load and make subruns together
        assert np.unique(sum_super["sum"]).size == 1
        # superruns in only-combining mode will load and make subruns separately
        assert np.unique(_sum_super["sum"]).size != 1

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
        if os.path.exists(self.tempdir2):
            shutil.rmtree(self.tempdir2)

    def _create_subruns(self, n_subruns=3):
        self.now = datetime.datetime.now()
        self.now.replace(tzinfo=pytz.utc)
        self.subrun_ids = [str(r) for r in range(n_subruns)]

        for run_id in self.subrun_ids:
            rr = self.context.get_array(run_id, "records")
            rg = self.context.get_array(run_id, "ranges")
            assert np.min(rr["time"]) == np.min(rg["time"])
            assert np.max(strax.endtime(rr)) == np.max(strax.endtime(rg))
            time = np.min(rg["time"])
            endtime = np.max(strax.endtime(rg))
            self.context.set_config(
                {"secret_time_offset": int(endtime + self.offset_between_subruns)}
            )
            self._write_run_doc(
                run_id,
                self.now + datetime.timedelta(0, int(time)),
                self.now + datetime.timedelta(0, int(endtime)),
            )
            assert self.context.is_stored(run_id, "records")
            assert self.context.is_stored(run_id, "ranges")

    def _write_run_doc(self, run_id, time, endtime):
        """Function which writes a dummy run document."""
        run_doc = {
            "name": run_id,
            "start": time,
            "end": endtime,
            "mode": self.subrun_modes[int(run_id) % 2],
            "source": self.subrun_source,
        }
        with open(self.context.storage[0]._run_meta_path(str(run_id)), "w") as fp:
            json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)


@strax.takes_config(
    strax.Option(
        name="some_additional_value",
        default=42,
        help="Some additional value for merger",
    )
)
class RecordsExtension(strax.Plugin):
    depends_on = "records"
    provides = "records_extension"
    dtype = strax.time_dt_fields + [(("Some additional field", "additional_field"), np.int16)]
    allow_superrun = True

    def compute(self, records):
        res = np.zeros(len(records), self.dtype)
        res["time"] = records["time"]
        res["length"] = records["length"]
        res["dt"] = records["dt"]
        res["additional_field"] = self.config["some_additional_value"]
        return res


@strax.takes_config(
    strax.Option(
        name="some_additional_peak_value",
        default=42,
        help="Some additional value for merger",
    )
)
class PeaksExtension(strax.Plugin):
    depends_on = "peaks"
    provides = "peaks_extension"
    save_when = strax.SaveWhen.EXPLICIT
    dtype = strax.time_dt_fields + [
        (("Some additional field", "some_additional_peak_field"), np.int16)
    ]
    allow_superrun = True

    def compute(self, peaks):
        res = np.zeros(len(peaks), self.dtype)
        res["time"] = peaks["time"]
        res["length"] = peaks["length"]
        res["dt"] = peaks["dt"]
        res["some_additional_peak_field"] = self.config["some_additional_peak_value"]
        return res


class PeaksExtensionCopy(PeaksExtension):
    depends_on = "peaks_extension"
    provides = "peaks_extension_copy"

    def compute(self, peaks):
        return peaks


@strax.takes_config(
    strax.Option("secret_time_offset", type=int, default=0, track=False),
    strax.Option("n_chunks", type=int, default=10, track=False),
    strax.Option("chunks_length", type=int, default=1, track=True),
)
class Ranges(Plugin):
    provides = "ranges"
    depends_on: tuple = tuple()
    rechunk_on_save = False
    dtype = [
        (("Numbers in order", "data"), np.int32),
    ] + strax.time_dt_fields

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config["n_chunks"]

    def compute(self, chunk_i):
        length = self.config["chunks_length"]
        r = np.zeros(length, self.dtype)
        t0 = chunk_i + self.config["secret_time_offset"]
        r["time"] = t0 + np.arange(length)
        r["length"] = r["dt"] = 1
        r["data"] = r["time"]
        return self.chunk(start=t0, end=t0 + length, data=r)


class Sum(ExhaustPlugin):
    provides = "sum"
    depends_on = "ranges"
    dtype = [
        (("Sum of numbers", "sum"), np.int32),
    ] + strax.time_dt_fields
    save_when = strax.SaveWhen.EXPLICIT
    allow_superrun = True

    def compute(self, ranges):
        s = np.zeros(len(ranges), self.dtype)
        s["time"] = ranges["time"]
        s["length"] = ranges["length"]
        s["dt"] = ranges["dt"]
        # if data is in order, and uniform, sum will only have one value
        s["sum"] = ranges["data"] + ranges["data"][::-1]
        return s
