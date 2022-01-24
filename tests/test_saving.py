import unittest
import strax
from strax.testutils import Records, Peaks
import os
import shutil
import tempfile


class TestPerRunDefaults(unittest.TestCase):
    """Test the saving behavior of the context"""
    def setUp(self):
        self.test_run_id = '0'
        self.target = 'records'
        self.path = os.path.join(tempfile.gettempdir(), 'strax_data')
        self.st = strax.Context(use_per_run_defaults=True,
                                register=[Records],
                                storage=[strax.DataDirectory(self.path)])
        assert not self.st.is_stored(self.test_run_id, self.target)

    def tearDown(self):
        if os.path.exists(self.path):
            print(f'rm {self.path}')
            shutil.rmtree(self.path)

    def test_savewhen_never(self, **kwargs):
        self.set_save_when('NEVER')
        self.st.make(self.test_run_id, self.target, **kwargs)
        assert not self.is_stored()

    def test_savewhen_never_with_save(self):
        should_fail_with_save = self.test_savewhen_never
        self.assertRaises(ValueError, should_fail_with_save, save=self.target)

    def test_savewhen_explict_without_save(self):
        self.set_save_when('EXPLICIT')
        self.st.make(self.test_run_id, self.target)
        assert not self.is_stored()

    def test_savewhen_explict_with_save(self):
        self.set_save_when('EXPLICIT')
        self.st.make(self.test_run_id, self.target, save=self.target)
        assert self.is_stored()

    def test_savewhen_target(self):
        self.set_save_when('TARGET')
        self.st.make(self.test_run_id, self.target)
        assert self.is_stored()

    def test_savewhen_always(self):
        self.set_save_when('ALWAYS')
        self.st.make(self.test_run_id, self.target)
        assert self.is_stored()

    def is_stored(self):
        return self.st.is_stored(self.test_run_id, self.target)

    def set_save_when(self, mode: str):
        if not hasattr(strax.SaveWhen, mode.upper()):
            raise ValueError(f'No such saving mode {mode}')
        save_mode = getattr(strax.SaveWhen, mode.upper())
        self.st._plugin_class_registry[self.target].save_when = save_mode

    def test_raise_corruption(self):
        self.set_save_when('ALWAYS')
        self.st.make(self.test_run_id, self.target)
        assert self.is_stored()
        storage = self.st.storage[0]
        data_key = self.st.key_for(self.test_run_id, self.target)
        data_path = os.path.join(storage.path, str(data_key))
        assert os.path.exists(data_path)
        metadata = storage.backends[0].get_metadata(data_path)
        assert isinstance(metadata, dict)

        # copied from FileSytemBackend (maybe abstractify the method separately?)
        prefix = strax.dirname_to_prefix(data_path)
        metadata_json = f'{prefix}-metadata.json'
        md_path = os.path.join(data_path, metadata_json)
        assert os.path.exists(md_path)

        # Corrupt the metadata (making it non-JSON parsable)
        md_file = open(md_path, 'a')
        # Append 'hello' at the end of file
        md_file.write('Adding a non-JSON line to the file to corrupt the metadata')
        # Close the file
        md_file.close()

        # Now we should get an error since the metadata data is corrupted
        with self.assertRaises(strax.DataCorrupted):
            self.st.get_array(self.test_run_id, self.target)

        # Also test the error is raised if be build a target that depends on corrupted data
        self.st.register(Peaks)
        with self.assertRaises(strax.DataCorrupted):
            self.st.get_array(self.test_run_id, 'peaks')

        # Cleanup if someone wants to re-use this self.st
        del self.st._plugin_class_registry['peaks']
