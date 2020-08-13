from strax.testutils import *
import pytest


def test_child_plugin_config():
    """
    Test if options of parent plugin are correctly replaced by config
    options of the child plugin. The following things are test:

    1. Default values are set correctly.
    2. set_config works properly
    3. Execution order does not change the result e.g. parent is changed
        if child is executed first.

    Following kind of settings are tested:

    1. Tracked options which are specified via the context, e.g.
        n_pmts_tpc.
    2. Untracked options which are specified via the context, e.g.
        channel_map.
    3. Settings only specified in the parent and therefore shared by the
        child.
    4. Settings only specified for the children but not for the parent.
    5. Settings which are used in both plugins, but for which the child
        uses a different value than the parent (child_options)
    """
    mystrax = strax.Context(storage=[],
                            config={'context_option': DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'][0],
                                    'more_special_context_option': immutabledict(tpc=DEFAULT_CONFIG_TEST['channel_map_parent'])
                                    },
                            register=[Records, Peaks, ParentPlugin, ChildPlugin],
                            allow_multiprocess=True)

    parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')
    child = mystrax.get_array(run_id=run_id, targets='peaks_child')

    # ---------------------------------
    # Checking if default setting of
    # parent and child are correct:
    # ---------------------------------
    # Unchanged values and common values::
    assert np.all(parent['time'] == child['time']), 'Time field is different for parent and child'
    assert np.all(parent['length'] == child['length']), 'Length field is different for parent and child'
    assert np.all(parent['dt'] == child['dt']), 'dt field is different for parent and child'

    m = np.all(parent['max_gap'] == child['max_gap'])
    assert m, 'parent_unique_option is not equal in parent and child'

    # Values which should have changed:
    t = np.all(parent['area'] == DEFAULT_CONFIG_TEST['area_parent'])
    assert t, 'by_child_overwrite_option has changed the value of the parent'
    t = np.all(child['area'] == DEFAULT_CONFIG_TEST['area_child'])
    assert t, 'by_child_overwrite_option has not been updated for the child'

    t = np.all(parent['n_hits'] == DEFAULT_CONFIG_TEST['nhits_parent'])
    assert t, '2nd_child_exclusive_option_child changed parent wtf?'

    t = np.all(child['n_hits'] == DEFAULT_CONFIG_TEST['nhits_child'])
    assert t, '2nd_child_exclusive_option_child got the wrong value'

    # Checking the shapes and values:
    mes = ('The channel identfication did not work for the parent',
           'This means "more_special_context_option_child" did not work properly')
    t = np.all(parent['area_per_channel'] == DEFAULT_CONFIG_TEST['area_per_channel_both'])
    assert t, mes
    mes = ('Parent array does not have the correct shape.'
           ' This means "context_option" was wrong.')
    assert parent[0]['area_per_channel'].shape == DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'], mes

    mes = ('The channel identfication did not work for the child',
           'This means "more_special_context_option_child" did not work properly')
    sc, ec = DEFAULT_CONFIG_TEST['channel_map_child']
    t = np.all(child['area_per_channel'][:, sc: ec] == DEFAULT_CONFIG_TEST['area_per_channel_both'])
    t = t & np.all(child['area_per_channel'][:, :sc] == 0)
    assert t, mes
    mes = ('Child array does not have the correct shape.'
           ' This means "context_option" was wrong.')
    assert child[0]['area_per_channel'].shape == DEFAULT_CONFIG_TEST['area_per_channel_shape_child'], mes

    # --------------------------------
    # Checking if set_config() changes
    # are propagated correctly.
    # --------------------------------
    new_test_settings = {'by_child_overwrite_option': 22,
                        'by_child_overwrite_option_child': 16,
                        'parent_unique_option': 99,
                        'context_option_child': 35,
                        }
    mystrax.set_config(new_test_settings)

    # Normal order
    parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')
    child = mystrax.get_array(run_id=run_id, targets='peaks_child')
    prefix = 'In the test of set_config in the normal order (first parent then child) '

    t = np.all(parent['area'] == new_test_settings['by_child_overwrite_option'])
    assert t, prefix + 'by_child_overwrite_option has changed the value of the parent'

    t = np.all(child['area'] == new_test_settings['by_child_overwrite_option_child'])
    assert t, prefix + 'by_child_overwrite_option has not been updated for the child'

    m = np.all((parent['max_gap'] == new_test_settings['parent_unique_option'])
               & (child['max_gap'] == new_test_settings['parent_unique_option']))
    assert m, prefix + 'parent_unique_option is not equal in parent and child'

    mes = (prefix,
           'Parent array does not have the correct shape.'
           ' This means "context_option" was wrong. '
           'Although we changed only the child.'
           )
    assert parent[0]['area_per_channel'].shape == DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'], mes

    mes = (prefix,
           'Child array does not have the correct shape.'
           ' This means "context_option" was wrong.')
    assert child[0]['area_per_channel'].shape == (new_test_settings['context_option_child'],), mes

    # Make sure reversed order does not overwrite anything:
    child = mystrax.get_array(run_id=run_id, targets='peaks_child')
    parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')
    prefix = 'In the test of set_config in the inverted order (first child then parent)'

    t = np.all(parent['area'] == new_test_settings['by_child_overwrite_option'])
    assert t, prefix + 'by_child_overwrite_option has changed the value of the parent'

    t = np.all(child['area'] == new_test_settings['by_child_overwrite_option_child'])
    assert t, prefix + 'by_child_overwrite_option has not been updated for the child'

    m = np.all((parent['max_gap'] == new_test_settings['parent_unique_option'])
               & (child['max_gap'] == new_test_settings['parent_unique_option']))
    assert m, prefix + 'parent_unique_option is not equal in parent and child'

    mes = (prefix,
           'Parent array does not have the correct shape.'
           ' This means "context_option" was wrong. '
           'Although we changed only the child.'
           )
    assert parent[0]['area_per_channel'].shape == DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'], mes

    mes = (prefix,
           'Child array does not have the correct shape.'
           ' This means "context_option" was wrong.')
    assert child[0]['area_per_channel'].shape == (new_test_settings['context_option_child'],), mes

def test_child_plugin_lienage():
    """
    Similar test as above, but this time week check the lineage/hash of
    the child and parent.
    """
    mystrax = strax.Context(storage=[],
                            config={'context_option': DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'][0],
                                    'more_special_context_option': immutabledict(tpc=DEFAULT_CONFIG_TEST['channel_map_parent'])
                                    },
                            register=[Records, Peaks, ParentPlugin, ChildPlugin],
                            allow_multiprocess=True)

    def _get_hashes():
        hash_child = mystrax.key_for('0', 'peaks_child').lineage_hash
        hash_parent = mystrax.key_for('0', 'peaks_parent').lineage_hash
        return hash_parent, hash_child

    def _check_hash(name, hash_parent, hash_child, changes_parent, changes_child):
        hp, hc = _get_hashes()
        if changes_parent:
            assert hp != hash_parent, f'{name} did not changed lineage of the parent!'
        else:
            assert hp == hash_parent, f'{name} changed lineage of the parent!'

        if changes_child:
            assert hc != hash_child, f'{name} did not changed lineage of the parent!'
        else:
            assert hc == hash_child, f'{name} changed lineage of the parent!'

        return hp, hc

    # -----------------------------
    # Checking if lineage of child and
    # parent are correct for defaults:
    # -----------------------------
    # Parent:
    lineage_parent = mystrax.key_for('0', 'peaks_parent').lineage['peaks_parent']
    true_config_parent = {'context_option': DEFAULT_CONFIG_TEST['area_per_channel_shape_parent'][0],
                          'by_child_overwrite_option': DEFAULT_CONFIG_TEST['area_parent'],
                          'parent_unique_option': DEFAULT_CONFIG_TEST['max_gap_both']}

    parent_name, parent_version, config = lineage_parent
    assert parent_name == 'ParentPlugin', 'Wrong parent name'
    assert parent_version == '0.0.5', 'Wrong parent version'
    for k, v in true_config_parent.items():
        assert config[k] == v, f'Parent has a wrong value for option: {k}'

    # Child:
    lineage_child = mystrax.key_for('0', 'peaks_child').lineage['peaks_child']
    true_config_child = {'parent_unique_option': DEFAULT_CONFIG_TEST['max_gap_both'],
                         'by_child_overwrite_option_child': DEFAULT_CONFIG_TEST['area_child'],
                         'context_option_child': DEFAULT_CONFIG_TEST['area_per_channel_shape_child'][0],
                         'child_exclusive_option': DEFAULT_CONFIG_TEST['nhits_child'],
                         'ParentPlugin': '0.0.5'}

    child_name, child_version, config = lineage_child
    assert child_name == 'ChildPlugin', 'Wrong child name'
    assert child_version == '0.0.1', 'Wrong child version'
    for k, v in true_config_child.items():
        assert config[k] == v, f'Parent has a wrong value for option: {k}'

    # -----------------------------
    # Last test, checking if set_config
    # changes the hash as expected:
    # -----------------------------
    # Child exclusive option:
    hp, hc = _get_hashes()
    mystrax.set_config({'child_exclusive_option': 8})
    hp, hc = _check_hash('child_exclusive_option',
                         hp, hc,
                         False, True
                         )

    # Parent exclusive option:
    mystrax.set_config({'by_child_overwrite_option': 42})
    hp, hc = _check_hash('parent_exclusive_option',
                         hp, hc,
                         True, False
                         )

    # Parent unique option (shared by child and arent):
    mystrax.set_config({'parent_unique_option': 33})
    hp, hc = _check_hash('child_exclusive_option',
                         hp, hc,
                         True, True
                         )

    # Context option parent:
    mystrax.set_config({'context_option': 33})
    hp, hc = _check_hash('child_exclusive_option',
                         hp, hc,
                         True, False
                         )

    # Context option child:
    mystrax.set_config({'context_option_child': 17})
    hp, hc = _check_hash('child_exclusive_option',
                         hp, hc,
                         False, True
                         )
