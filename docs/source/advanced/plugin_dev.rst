Plugin development
===================


Specifying options
-------------------

...

If an option is not set in the configuration, strax will use its default value if one is specified. If the plugin specifies no default, you will get a RuntimeError when attempting to start processing.

Even when a default is used, the value used will be stored in the metadata of the produced data. Thus, if you only change a default value for a plugin's option, you do NOT have to increment the plugin version to ensure data is rebuilt when needed.

Specify defaults in several ways:

- ``default``: Use the given value as default.
- ``default_factory``: Call the given function (with no arguments) to produce a default. Use for mutable values such as lists.
- ``default_per_run``: Specify a list of 2-tuples: ``(start_run, default)``. Here start_run is a numerized run name (e.g 170118_1327; note the underscore is valid in integers since python 3.6) and ``default`` the option that applies from that run onwards.

...
