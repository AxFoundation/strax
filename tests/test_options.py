"""
Test if options can be overwritten in the correct way
https://github.com/AxFoundation/strax/pull/392
"""
import strax
import numpy as np

# Initialize. We test both dt time-fields and time time-field
_dtype_name = 'variable'
_dtype = ('variable 1', _dtype_name)
test_dtype = [(_dtype, np.float64)] + strax.time_fields


def test_overwrite():
    @strax.takes_config(
        strax.Option('option',
                     default=False),
    )
    class BasePlugin(strax.Plugin):
        """The plugin that we will be sub-classing"""
        provides = 'base'
        dtype = test_dtype
        provides = 'base'
        depends_on = tuple()

        def compute(self, something):
            return np.ones(len(something), dtype=self.dtype)

    st = strax.Context(storage=[])
    st.register(BasePlugin)
    # Keep an account of this lineage hash such that we can compare it later
    lineage_base = st.key_for('0', 'base').lineage_hash

    try:
        @strax.takes_config(
            strax.Option('option',
                         default=True),
        )
        class CrashPlugin(BasePlugin):
            """
            Try subclassing with a different option default will cause a
            runtime error
            """
            pass

        st.register(CrashPlugin)

    except RuntimeError:
        print('Ran into a RuntimeError because we tried specifying an '
              'option twice. This is exactly what we want!')

    @strax.takes_config(
        strax.Option('option',
                     default=True,
                     overwrite=True),
    )
    class OverWritePlugin(BasePlugin):
        """Only overwrite the option, the rest is the same"""
        pass

    st.register(OverWritePlugin)

    assert st.key_for('0', 'base').lineage_hash != lineage_base, 'Lineage did not change'
    p = st.get_single_plugin('0', 'base')
    assert p.__class__.__name__ == 'OverWritePlugin'
    assert p.config['option'] is True, f'Option was not overwritten: {p.config}'
