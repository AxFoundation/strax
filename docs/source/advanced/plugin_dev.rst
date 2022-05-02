Plugin development
===================

Special time fields
-----------------------
The ``time``, ``endtime``, ``dt`` and ``length`` fields have special meaning for strax.

It is useful for most plugins to output a ``time`` and ``endtime`` field, indicating the
start and (exclusive) end time of the entities you are producing.
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


Plugin types
----------------------

There are several plugin types:
   * ``Plugin``: The general type of plugin. Should contain at least ``depends_on = <datakind>``, ``provides = <datatype>``, ``def compute(self, <datakind>)``, and ``dtype = <dtype>`` or ``def infer_dtype(): <>``.
   * ``OverlapWindowPlugin``: Allows a plugin to look for data in adjacent chunks. A ``OverlapWindowPlugin`` assumes all inputs are sorted by *endtime*. This only works for disjoint intervals such as peaks or events, but NOT records! The user has to define ``get_window_size(self)`` along with the plugin which returns the required chunk extension in nanoseconds.
   * ``LoopPlugin``: Allows user to loop over a given datakind and find the corresponding data of a lower datakind using for example `def compute_loop(self, events, peaks)` where we loop over events and get the corresponding peaks that are within the time range of the event. By default the second argument (``peaks``) must be fully contained in the first argument (``events`` ). If a touching time window is desired set the class attribute ``time_selection`` to `'`touching'``.
   * ``CutPlugin``: Plugin type where using ``def cut_by(self, <datakind>)`` inside the plugin a user can return a boolean array that can be used to select data.
   * ``MergeOnlyPlugin``: This is for internal use and only merges two plugins into a new one. See as an example in straxen the ``EventInfo`` plugin where the following datatypes are merged ``'events', 'event_basics', 'event_positions', 'corrected_areas', 'energy_estimates'``.
   * ``ParallelSourcePlugin``: For internal use only to parallelize the processing of low level plugins. This can be activated using stating ``parallel = 'process'`` in a plugin.


Minimal examples
----------------------
Below, each of the plugins is minimally worked out, each plugin can be worked
out into much greater detail, see e.g. the
`plugins in straxen <https://github.com/XENONnT/straxen/tree/master/straxen/plugins>`_.


strax.Plugin
____________
.. code-block:: python

    # To tests, one can use these dummy Peaks and Records from strax
    import strax
    import numpy as np
    from strax.testutils import Records, Peaks, run_id
    st = strax.Context(register=[Records, Peaks])

    class BasePlugin(strax.Plugin):
        """The most common plugin where computations on data are performed in strax"""
        depends_on = 'records'

        # For good practice always specify the version and provide argument
        provides = 'simple_data'
        # When you are developing a new plugin, you can set this version to None to
        # auto-infer this from the code of the plugin.
        __version__ = '0.0.0'

        # We need to specify the datatype, for this example, we are
        # going to calculate some areas
        dtype = strax.time_fields + [(("Total ADC counts",'area'), np.int32)]

        def compute(self, records):
            result = np.zeros(len(records), dtype=self.dtype)

            # All data in strax must have some sort of time fields
            result['time'] = records['time']
            result['endtime'] = strax.endtime(records)

            # For this example, we calculate the total sum of the records-data
            result['area'] = np.sum(records['data'], axis = 1)
            return result

    st.register(BasePlugin)
    st.get_df(run_id, 'simple_data')


strax.OverlapWindowPlugin
_________________________
.. code-block:: python

    class OverlapPlugin(strax.OverlapWindowPlugin):
        """
        Allow peaks get_window_size() left and right to get peaks
            within the time range
        """
        depends_on = 'peaks'
        provides = 'overlap_data'

        dtype = strax.time_fields + [(("total peaks", 'n_peaks'), np.int16)]

        def get_window_size(self):
            # Look 10 ns left and right of each peak
            return 10

        def compute(self, peaks):
            result = np.zeros(1, dtype=self.dtype)
            result['time'] = np.min(peaks['time'])
            result['endtime'] = np.max(strax.endtime(peaks))
            result['n_peaks'] = len(peaks)
            return result

    st.register(OverlapPlugin)
    st.get_df(run_id, 'overlap_data')


strax.LoopPlugin
__________
.. code-block:: python

    class LoopData(strax.LoopPlugin):
        """Loop over peaks and find the records within each of those peaks."""
        depends_on = 'peaks', 'records'
        provides = 'looped_data'

        dtype = strax.time_fields + [(("total records", 'n_records'), np.int16)]

        # The LoopPlugin specific requirements
        time_selection = 'fully_contained' # other option is 'touching'
        loop_over = 'peaks'

        # Use the compute_loop() instead of compute()
        def compute_loop(self, peaks, records):
            result = np.zeros(len(peaks), dtype=self.dtype)
            result['time'] = np.min(peaks['time'])
            result['endtime'] = np.max(strax.endtime(peaks))
            result['n_records'] = len(records)
            return result
    st.register(LoopData)
    st.get_df(run_id, 'looped_data')


strax.CutPlugin
_________________________
.. code-block:: python

    class CutData(strax.CutPlugin):
        """
        Create a boolean array if an entry passes a given cut,
            in this case if the peak has a positive area
        """
        depends_on = 'peaks'
        provides = 'cut_data'

        # Use cut_by() instead of compute() to generate a boolean array
        def cut_by(self, peaks):
            return peaks['area']>0

    st.register(CutData)
    st.get_df(run_id, 'cut_data')


strax.MergeOnlyPlugin
________
.. code-block:: python

    class MergeData(strax.MergeOnlyPlugin):
        """Merge datatypes of the same datakind into a single datatype"""
        depends_on = ('peaks', 'cut_data')
        provides = 'merged_data'

        # You only need specify the dependencies, those are merged.

    st.register(MergeData)
    st.get_array(run_id, 'merged_data')


Plugin inheritance
----------------------
It is possible to inherit the ``compute()`` method of an already existing plugin with another plugin. We call these types of plugins child plugins. Child plugins are recognized by strax when the ``child_plugin`` attribute of the plugin is set to ``True``. Below you can find a simple example of a child plugin with its parent plugin:

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

The ``super().compute()`` statement in the ``compute`` method of ``ChildPlugin`` allows us to execute the code of the parent's compute method without duplicating it. Additionally, if needed, we can extend the code with some for the child-plugin unique computation steps.

To allow for the child plugin to have different settings then its parent (e.g. ``'by_child_overwrite_option'`` in ``self.config['by_child_overwrite_option']`` of the parent's ``compute`` method), we have to use specific child option. These options will be recognized by strax and overwrite the config values of the parent parameter during the initialization of the child-plugin. Hence, these changes only affect the child, but not the parent.

An option can be flagged as a child option if the corresponding option attribute is set ``child_option=True``. Further, the option name which should be overwritten must be specified via the option attribute ``parent_option_name``.

The lineage of a child plugin contains in addition to its options the name and version of the parent plugin.
