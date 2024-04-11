from strax.testutils import RecordsWithTimeStructure, DownSampleRecords, run_id
import strax
import numpy as np

import os
import tempfile
import shutil
import uuid
import unittest


class TestContext(unittest.TestCase):
    """Tests for DownChunkPlugin class."""

    def setUp(self):
        """Make temp folder to write data to."""
        temp_folder = uuid.uuid4().hex
        self.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        assert not os.path.exists(self.tempdir)

    def tearDown(self):
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_down_chunking(self):
        st = self.get_context()
        st.register(RecordsWithTimeStructure)
        st.register(DownSampleRecords)

        st.make(run_id, "records")
        st.make(run_id, "records_down_chunked")

        chunks_records = st.get_meta(run_id, "records")["chunks"]
        chunks_records_down_chunked = st.get_meta(run_id, "records_down_chunked")["chunks"]

        _chunks_are_downsampled = len(chunks_records) * 2 == len(chunks_records_down_chunked)
        assert _chunks_are_downsampled

        _chunks_are_continues = np.all(
            [
                chunks_records_down_chunked[i]["end"] == chunks_records_down_chunked[i + 1]["start"]
                for i in range(len(chunks_records_down_chunked) - 1)
            ]
        )
        assert _chunks_are_continues

    def test_down_chunking_multi_processing(self):
        st = self.get_context(allow_multiprocess=True)
        st.register(RecordsWithTimeStructure)
        st.register(DownSampleRecords)

        st.make(run_id, "records", max_workers=1)

        class TestMultiProcessing(DownSampleRecords):
            parallel = True

        st.register(TestMultiProcessing)
        with self.assertRaises(NotImplementedError):
            st.make(run_id, "records_down_chunked", max_workers=2)

    def get_context(self, **kwargs):
        """Simple context to run tests."""
        st = strax.Context(storage=self.get_mock_sf(), check_available=("records",), **kwargs)
        return st

    def get_mock_sf(self):
        mock_rundb = [{"name": "0", strax.RUN_DEFAULTS_KEY: dict(base_area=43)}]
        sf = strax.DataDirectory(path=self.tempdir, deep_scan=True, provide_run_metadata=True)
        for d in mock_rundb:
            sf.write_run_metadata(d["name"], d)
        return sf
