import strax
import numpy as np
import tempfile
import unittest
import hypothesis
from hypothesis import given


@strax.takes_config(
    strax.Option(name='int_option', type=int, default=42),
    strax.Option(name='str_option', type=str, default='forty_two'),
    strax.Config(name='mixed', type=int, default=42),
)
class DummyPlugin(strax.Plugin):
    depends_on = ()
    provides = ('dummy_data',)
    dtype = strax.dtypes.time_fields + [(('Some data description', 'some_data_name'), np.int32),]

    int_config = strax.Config(type=int, default=42)
    str_config = strax.Config(type=str, default='forty_two')


class TestPluginConfig(unittest.TestCase):
    @staticmethod
    def get_plugin(config):
        with tempfile.TemporaryDirectory() as temp_dir:
            context = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                        deep_scan=True),
                            config=config,
                            register=[DummyPlugin],
                            use_per_run_defaults=True,
                            )
                            
            return context.get_single_plugin('321', 'dummy_data')

    def test_config_defaults(self):
        p = self.get_plugin({})
        assert p.int_config == p.int_option == 42    
        assert p.str_option == p.str_config == 'forty_two'

    @given(
        hypothesis.strategies.integers(),
        hypothesis.strategies.text(),
    )
    def test_config_attr_access(self, int_value, str_value):
        config = { 
            'int_config': int_value,
            'str_config': str_value,
            'int_option': int_value,
            'str_option': str_value,
            }
        p = self.get_plugin(config)
    
        assert p.int_config == p.int_option == int_value    
        assert p.str_option == p.str_config == str_value

    @given(
        hypothesis.strategies.integers(),
        hypothesis.strategies.text(),
    )
    def test_config_dict_access(self, int_value, str_value):
        '''
        Test backward compatibility
        '''
        config = { 
            'int_config': int_value,
            'str_config': str_value,
            'int_option': int_value,
            'str_option': str_value,
            }

        p = self.get_plugin(config)
        assert p.config['int_config'] == p.config['int_option'] == int_value    
        assert p.config['str_config'] == p.config['str_option'] == str_value

    def test_config_backward_compatibility(self):
        p = self.get_plugin({})
        assert p.mixed == 42
