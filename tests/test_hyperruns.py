import os
import re
import json
import datetime
import pytz
from bson import json_util
import tempfile
import unittest
import shutil
import numpy as np
import strax
from strax import Plugin, ExhaustPlugin


@strax.takes_config(
    strax.Option("n_chunks", type=int, default=1, track=True),
    strax.Option("chunks_length", type=int, default=10, track=True),
    strax.Option("zero_offset", type=int, default=0, track=False),
)
class Ranges(Plugin):
    provides = "ranges"
    depends_on: tuple = tuple()
    dtype = [
        (("Numbers in order", "data"), np.int32),
    ] + strax.time_dt_fields

    rechunk_on_save = False

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config["n_chunks"]

    def compute(self, chunk_i):
        length = self.config["chunks_length"]
        r = np.zeros(length, self.dtype)
        t0 = self.config["zero_offset"]
        r["time"] = np.arange(length) + t0
        r["length"] = r["dt"] = 1
        r["data"] = r["time"]
        return self.chunk(start=t0, end=t0 + length, data=r)


class Sum(ExhaustPlugin):
    provides = "sum"
    depends_on = "ranges"
    dtype = [
        (("Sum of numbers", "sum"), np.int32),
    ] + strax.time_dt_fields
    allow_hyperrun = True

    def compute(self, ranges):
        s = np.zeros(len(ranges), self.dtype)
        s["time"] = ranges["time"]
        s["length"] = ranges["length"]
        s["dt"] = ranges["dt"]
        # if data is in order, and uniform, sum will only have one value
        s["sum"] = ranges["data"] + ranges["data"][::-1]
        return s


class TestHyperRuns(unittest.TestCase):
    def setUp(self):
        self.offset_between_subruns = 10
        self.superrun_name = "_000000"
        self.hyperrun_name = "__000000"
        # Temp directory for storing record data for the tests.
        # Will be removed during TearDown.
        self.tempdir = tempfile.mkdtemp()
        # with two storage frontends
        self.context = strax.Context(
            storage=[
                strax.DataDirectory(
                    self.tempdir, provide_run_metadata=True, readonly=False, deep_scan=True
                )
            ],
            register=[Ranges, Sum],
            config={
                "n_chunks": 1,
                "chunks_length": 10,
            },
        )
        self.context.set_context_config({"write_superruns": True, "use_per_run_defaults": False})

        logger = self.context.log

        lambda s: not logger.addFilter(
            re.match(".*Could not estimate run start and end time.*", s.getMessage())
        )
        self.context._plugin_class_registry["ranges"].chunk_target_size_mb = 1
        self._create_subruns()
        self.context.define_run(self.superrun_name, data=self.subrun_ids)  # Define superrun
        self.context.define_run(self.hyperrun_name, data=self.subrun_ids)  # Define hyperrun

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def _create_subruns(self, n_subruns=3):
        self.now = datetime.datetime.now()
        self.now.replace(tzinfo=pytz.utc)
        self.subrun_ids = [str(r) for r in range(n_subruns)]

        for run_id in self.subrun_ids:
            rg = self.context.get_array(run_id, "ranges")
            time = np.min(rg["time"])
            endtime = np.max(strax.endtime(rg))

            self._write_run_doc(
                run_id,
                self.now + datetime.timedelta(0, int(time)),
                self.now + datetime.timedelta(0, int(endtime)),
            )

            self.context.set_config(
                {"zero_offset": (int(run_id) + 1) * self.context.config["chunks_length"]}
            )
            assert self.context.is_stored(run_id, "ranges")

    def _write_run_doc(self, run_id, time, endtime):
        """Function which writes a dummy run document."""
        run_doc = {
            "name": run_id,
            "start": time,
            "end": endtime,
        }
        with open(self.context.storage[0]._run_meta_path(str(run_id)), "w") as fp:
            json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)

    def test_load_superruns_and_hyperruns(self):
        """Test loading superruns and hyperruns.

        The test also shows the difference between the two.

        """
        sum_super = self.context.get_array(self.superrun_name, "sum")
        sum_hyper = self.context.get_array(self.hyperrun_name, "sum")

        # superruns will still load and make subruns separately
        assert np.unique(sum_super["sum"]).size != 1
        # hyperruns will load and make subruns together
        assert np.unique(sum_hyper["sum"]).size == 1
