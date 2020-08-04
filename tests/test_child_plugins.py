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

    # Special UserWarning which will be raised, but can be "ignored" since it is just a hint for the user
    warn_message = ('You specified plugin ChildPlugin as a child plugin. Found the option '
                    '2nd_child_exclusive_option_child with the ending _child which was '
                    'not specified as a child option. Was this intended?')
    with pytest.warns(UserWarning, match=warn_message):
        mystrax = strax.Context(storage=[],
                                config={'context_option': 4,
                                        'more_special_context_option': immutabledict(tpc=(0, 4))
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
        assert np.all(parent['area'] == 2), 'by_child_overwrite_option has changed the value of the parent'
        assert np.all(child['area'] == 4), 'by_child_overwrite_option has not been updated for the child'

        assert np.all(parent['n_hits'] == 0), '2nd_child_exclusive_option_child changed parent wtf?'
        assert np.all(child['n_hits'] == 2), '2nd_child_exclusive_option_child got the wrong value'

        # Checking the shapes and values:
        mes = ('The channel identfication did not work for the parent',
               'This means "more_special_context_option_child" did not work properly')
        assert np.all(parent['area_per_channel'] == 1), mes
        mes = ('Parent array does not have the correct shape.'
               ' This means "context_option" was wrong.')
        assert parent[0]['area_per_channel'].shape == (4,), mes

        mes = ('The channel identfication did not work for the child',
               'This means "more_special_context_option_child" did not work properly')
        assert np.all(child['area_per_channel'][:, 4:10] == 1), mes
        mes = ('Child array does not have the correct shape.'
               ' This means "context_option" was wrong.')
        assert child[0]['area_per_channel'].shape == (10,), mes

        # --------------------------------
        # Checking if set_config() changes
        # are propagated correctly.
        # --------------------------------
        mystrax.set_config({'by_child_overwrite_option': 22,
                            'by_child_overwrite_option_child': 16,
                            'parent_unique_option': 99,
                            'context_option_child': 35,
                            })

        # Normal order
        parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')
        child = mystrax.get_array(run_id=run_id, targets='peaks_child')

        prefix = 'In the test of set_config in the normal oder '
        assert np.all(parent['area'] == 22), prefix + 'by_child_overwrite_option has changed the value of the parent'
        assert np.all(child['area'] == 16), prefix + 'by_child_overwrite_option has not been updated for the child'

        m = np.all((parent['max_gap'] == 99) & (child['max_gap'] == 99))
        assert m, prefix + 'parent_unique_option is not equal in parent and child'

        mes = (prefix,
               'Parent array does not have the correct shape.'
               ' This means "context_option" was wrong.'
               )
        assert parent[0]['area_per_channel'].shape == (4,), mes

        mes = (prefix,
               'Child array does not have the correct shape.'
               ' This means "context_option" was wrong.')
        assert child[0]['area_per_channel'].shape == (35,), mes

        # Make sure reversed order does not overwrite anything:
        child = mystrax.get_array(run_id=run_id, targets='peaks_child')
        parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')

        prefix = 'In the test of set_config in the inverted oder '
        assert np.all(parent['area'] == 22), prefix + 'by_child_overwrite_option has changed the value of the parent'
        assert np.all(child['area'] == 16), prefix + 'by_child_overwrite_option has not been updated for the child'

        m = np.all((parent['max_gap'] == 99) & (child['max_gap'] == 99))
        assert m, prefix + 'parent_unique_option is not equal in parent and child'

        mes = (prefix,
               'Parent array does not have the correct shape.'
               ' This means "context_option" was wrong.'
               )
        assert parent[0]['area_per_channel'].shape == (4,), mes

        mes = (prefix,
               'Child array does not have the correct shape.'
               ' This means "context_option" was wrong.')
        assert child[0]['area_per_channel'].shape == (35,), mes

def test_child_plugin_lienage():
    """
    Similar test as above, but this time week check the lineage/hash of
    the child and parent.
    """
    # Special UserWarning which will be raised, but can be "ignored" since it is just a hint for the user
    warn_message = ('You specified plugin ChildPlugin as a child plugin. Found the option '
                    '2nd_child_exclusive_option_child with the ending _child which was '
                    'not specified as a child option. Was this intended?')
    with pytest.warns(UserWarning, match=warn_message):
        mystrax = strax.Context(storage=[],
                                config={'context_option': 4,
                                        'more_special_context_option': immutabledict(tpc=(0, 4))
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
        true_config_parent = {'context_option': 4,
                              'by_child_overwrite_option': 2,
                              'parent_unique_option': 10}

        parent_name, parent_version, config = lineage_parent
        assert parent_name == 'ParentPlugin', 'Wrong parent name'
        assert parent_version == '0.0.5', 'Wrong parent version'
        for k, v in true_config_parent.items():
            assert config[k] == v, f'Parent has a wrong value for context_option {k}'

        # Child:
        lineage_child = mystrax.key_for('0', 'peaks_child').lineage['peaks_child']
        true_config_child = {'parent_unique_option': 10,
                             'by_child_overwrite_option_child': 4,
                             'context_option_child': 10,
                             'child_exclusive_option': 6,
                             '2nd_child_exclusive_option_child': 2,
                             'ParentPlugin': '0.0.5'}

        child_name, child_version, config = lineage_child
        assert child_name == 'ChildPlugin', 'Wrong child name'
        assert child_version == '0.0.1', 'Wrong child version'
        for k, v in true_config_child.items():
            assert config[k] == v, f'Parent has a wrong value for context_option {k}'

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
