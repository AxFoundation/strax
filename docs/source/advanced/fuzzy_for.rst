Fuzzy for functionality
=======================
Since strax tracks lineages, updates to low level plugins may change the
availability of high level data. When a low level plugin is changed (for example
the version of a plugin is incremented), strax will recognize that the data corresponding
to the plugin whereof the version is changed is not stored (since only the
previous version is stored). This safeguards that the data that the user is loading
is always consistent with the context.

**This functionality can partially be disabled using fuzzy-for settings. This should
only be done temporarily or for quick checks as strax is not anymore checking if
the entire ancestry of the requested and the delivered data is consistent.**

When to use
-----------
There are situations where the above robustness of the context is not what the user
wants. Such situations can be if a user is developing a new plugin on the master
branch, when the master branch has some changes in the lower level plugins.
The user in this case cannot easily check if the plugin works on data, as no data
is available in the context of the master branch. In this case, the user might want
to tell the context to just load whatever data is available, ignoring changes in
a specific plugin. Another example would be if a dataset was simulated with specific
instructions and a user wants to quickly look at the data in the simulated dataset
without having to manually check which context was used for simulating this data
(of course, the best way to solve this would be to open the metadata that is stored
with the simulation files and construct the context from those options).

How to use
----------
There are two ways of ignoring the lineage. Both are set in the context config
(see context.context_config):
 - ``fuzzy_for_options`` a tuple of options to specify that each option with a
   name in the tuple can be ignored
 - ``fuzzy_for`` a tuple of data-types to ignore.

In the example below, we will use setting the ``fuzzy_for`` option. We will use
the online context from `straxen <http://github.com/XENONnT/straxen>`_ to illustrate
how the options are set.


.. code-block:: python

    import straxen
    # Use a context that can load data from a datatype 'peak-basics'
    st = straxen.contexts.xenonnt_online()
    run_id, target = '022880', 'peak_basics'

    # Check if the data is stored for this run and datatype
    print(f'{run_id} {target} is stored: {st.is_stored(run_id, target)}')

    # Now let's mimic the situation wherein the version of the plugin that provides
    # peak basics has changed (it has a different version). We will do so by changing
    # the version of the plugin below
    PeakBasics = st._plugin_class_registry[target]
    PeakBasics.__version__ = 'does_not_exist'
    print(f'{run_id} {target} is stored: {st.is_stored(run_id, target)}')

    # The print statement will tell us the data is not stored. To load the data
    # from the default version of PeakBasics we will use the fuzzy-for option:
    st.context_config['fuzzy_for'] = (target,)
    print(f'{run_id} {target} is stored: {st.is_stored(run_id, target)}')

The block above prints:

.. code-block::  bash

    022880 peak_basics is stored: True
    022880 peak_basics is stored: False
    022880 peak_basics is stored: True

Is it advisable / safe to use?
------------------------------
For running production analyses, one should never base results on a context where
fuzzy-ness is enabled.

For quick tests, it is save to use. If new data is made based on a fuzzy context,
this is not stored to prevent the creation of data-files with unreproducible
results.

Additionally (depending on the StorageFrontend), loading data with fuzzy options
will be generally much slower. For example, the most commonly used StorageFrontend,
the DataDirectory scans all folders within it's parent directory and filters the
meta-data in search for a folder with a lineage compatible with the fuzzy for
options.
