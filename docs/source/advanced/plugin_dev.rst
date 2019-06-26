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

