from strax.testutils import *


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
                            config={'context_option': 4,
                                    'more_special_context_option': immutabledict(tpc=(0, 4))
                                    },
                            register=[Records, Peaks, ParentPlugin, ChildPlugin],
                            allow_multiprocess=True)

    parent = mystrax.get_array(run_id=run_id, targets='peaks_parent')
    child = mystrax.get_array(run_id=run_id, targets='peaks_child')

    # Checking if default setting of parent and child are correct:
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

    # TODO Checking if set_config() changes are propagated correctly.

def test_child_plugin_lienage():
    pass