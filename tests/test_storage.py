import unittest
import strax
from strax.testutils import Records, Peaks
import os
import shutil
import tempfile


class TestPerRunDefaults(unittest.TestCase):
    """Test the saving behavior of the context"""
    def setUp(self):
        self.path = os.path.join(tempfile.gettempdir(), 'strax_data')
        self.st = strax.Context(use_per_run_defaults=True,
                                register=[Records],)
        self.target = 'records'

    def tearDown(self):
        if os.path.exists(self.path):
            print(f'rm {self.path}')
            shutil.rmtree(self.path)

    def test_write_data_dir(self):
        self.st.storage = [strax.DataDirectory(self.path)]
        run_id = '0'
        self.st.make(run_id, self.target)
        assert self.st.is_stored(run_id, self.target)

    def test_complain_run_id(self):
        self.st.storage = [strax.DataDirectory(self.path)]
        run_id = 'run-0'
        with self.assertRaises(ValueError):
            self.st.make(run_id, self.target)
