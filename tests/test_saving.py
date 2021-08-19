import unittest
import strax
from strax.testutils import Records
import os
import shutil
import tempfile


class TestPerRunDefaults(unittest.TestCase):
    """Test the saving behavior of the context"""
    def setUp(self):
        self.test_run_id = '0'
        self.target = 'records'
        self.path = os.path.join(tempfile.gettempdir(), 'strax_data')
        self.st = strax.Context(register=[Records],
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
