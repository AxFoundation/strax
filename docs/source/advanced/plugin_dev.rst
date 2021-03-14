Plugin development
===================

Special time fields
-----------------------
The ``time``, ``endtime``, ``dt`` and ``length`` fields have special meaning for strax.

It is useful for most plugins to output a ``time`` and ``endtime`` field, indicating the
start and (exclusive) end time of the entitities you are producing.
If you do not do this, your plugin cannot be loaded for part of a run (e.g. with ``seconds_range``).

Both ``time`` and ``endtime`` should be 64-bit integer timestamps in nanoseconds since the unix epoch. Instead of ``endtime``, you can provide ``dt`` (an integer time resolution in ns) and ``length`` (integer); strax will then compute the endtime as ``dt * length``. Lower-level datatypes commonly use this.

Usually, you simply pass these time-related fields through from one of your dependencies (e.g. ``events`` or ``peak_basics``). You should only modify them if you are changing data kind. If your plugin does defines a new data kind, you set the values yourself, depending on the thing you are making (events, peaks, etc).



Multiple outputs
------------------
Plugins can have multiple outputs. Do not use this if you just want to return a multi-column output (e.g. the area and width of a peak), that's what the structured arrays are for. But if you want to return incompatible kinds of rows, e.g. both records and hits, or peaks and some metadata for each chunk, multi-output support is essential.

To return multiple outputs from a plugin:
   * The `provides` tuple should have multiple elements, listing the provided data type names
   * The `dtype` and `data_kind` attributes should be dictionaries mapping data type names to dtypes and data kind names, respectively. The values of these dicts can be specified in the same way as the entire attribute would be for a single-output plugin
   * The `compute` method must return a dictionary mapping data types to results (structured numpy arrays or field/array dictionaries).


Options and defaults
----------------------

You can specify options using the `strax.takes_config` decorator and the `strax.Option` objects. See any plugin source code for example (todo: don't be lazy and explain).

There is a single configuration dictionary in a strax context, shared by all plugins. Be judicious in how you name your options to avoid clashes. "Threshold" is probably a bad name, "peak_min_channels" is better.

If an option is not set in the context's configuration, strax will use its default value if the option defines one. If the plugin specifies no default, you will get a RuntimeError when attempting to start processing.

Even when a default is used, the value used will be stored in the metadata of the produced data. Thus, if you only change a default value for a plugin's option, you do NOT have to increment the plugin version to ensure data is rebuilt when needed.

You can specify defaults in several ways:

- ``default``: Use the given value as default.
- ``default_factory``: Call the given function (with no arguments) to produce a default. Use for mutable values such as lists.
- ``default_per_run``: Specify a list of 2-tuples: ``(start_run, default)``. Here start_run is a numerized run name (e.g 170118_1327; note the underscore is valid in integers since python 3.6) and ``default`` the option that applies from that run onwards.
- The ``strax_defaults`` dictionary in the run metadata. This overrides any defaults specified in the plugin code, but take care -- if you change a value here, there will be no record anywhere of what value was used previously, so you cannot reproduce your results anymore!


Plugin types
----------------------

There are several plugin types:
   * `Plugin`: The general type of plugin. Should contain at least `depends_on = <datakind>`, `provides = <datatype>`, `def compute(self, <datakind>)`, and `dtype = <dtype> ` or `def infer_dtype(): <>`.
   * `OverlapWindowPlugin`: Allows a plugin to look for data in adjacent chunks. A OverlapWindowPlugin assumes: all inputs are sorted by *endtime*. This only works for disjoint intervals such as peaks or events, but NOT records! The user has to define get_window_size(self) along with the plugin which returns the required chunk extension in nanoseconds. 
   * `LoopPlugin`: Allows user to loop over a given datakind and find the corresponding data of a lower datakind using for example `def compute_loop(self, events, peaks)` where we loop over events and get the corresponding peaks that are within the time range of the event. Currently the second argument (`peaks`) must be fully contained in the first argument (`events` ).
   * `CutPlugin`: Plugin type where using `def cut_by(self, <datakind>)` inside the plugin a user can return a boolean array that can be used to select data.
   * `MergeOnlyPlugin`: This is for internal use and only merges two plugins into a new one. See as an example in straxen the `EventInfo` plugin where the following datatypes are merged `'events', 'event_basics', 'event_positions', 'corrected_areas', 'energy_estimates'`.
   * `ParallelSourcePlugin`: For internal use only to parallelize the processing of low level plugins. This can be activated using stating `parallel = 'process'` in a plugin.


Plugin inheritance
----------------------
It is possible to inherit the `compute()` method of an already existing plugin with another plugin. We call these types of plugins child plugins. Child plugins are recognized by strax when the `child_plugin` attribute of the plugin is set to `True`. Below you can find a simple example of a child plugin with its parent plugin:

.. code-block:: python

    @strax.takes_config(
    strax.Option('by_child_overwrite_option', type=int, default=5,
                 help="Option we will overwrite in our child plugin"),
    strax.Option('parent_unique_option', type=int, default=2,
                 help='Option which is not touched by the child and '
                      'therefore the same for parent and child'),
                      )
    class ParentPlugin(strax.Plugin):
        provides = 'odd_peaks'
        depends_on = 'peaks'
        __version__ = '0.0.1'
        dtype = parent_dtype

        def compute(self, peaks):
            peaks['area'] *= self.config['parent_unique_option']
            peaks['time'] *= self.config['by_child_overwrite_option']
            return res


    # Child:
    @strax.takes_config(
        strax.Option('by_child_overwrite_option_child',
                     default=3,
                     child_option=True,
                     parent_option_name='by_child_overwrite_option',
                     help="Option we will overwrite in our child plugin"),
        strax.Option('option_unique_child',
                      default=10,
                      help="Option we will overwrite in our child plugin"),
    )
    class ChildPlugin(ParentPlugin):
        provides = 'odd_peaks_child'
        depends_on = 'peaks'
        __version__ = '0.0.1'
        child_plugin = True

        def compute(self, peaks):
            res = super().compute(peaks)
            res['width'] = self.config['option_unique_child']
            return res

The `super().compute()` statement in the `compute` method of `ChildPlugin` allows us to execute the code of the parent's compute method without duplicating it. Additionally, if needed, we can extend the code with some for the child-plugin unique computation steps.

To allow for the child plugin to have different settings then its parent (e.g. `'by_child_overwrite_option'` in `self.config['by_child_overwrite_option']` of the parent's `compute` method), we have to use specific child option. These options will be recognized by strax and overwrite the config values of the parent parameter during the initialization of the child-plugin. Hence, these changes only affect the child, but not the parent.

An option can be flagged as a child option if the corresponding option attribute is set `child_option=True`. Further, the option name which should be overwritten must be specified via the option attribute `parent_option_name`.

The lineage of a child plugin contains in addition to its options the name and version of the parent plugin.
