Superruns
=========

Overview and motivation
------------------------
A superrun is a run defined by (parts of) other runs, which are called 'subruns'.
Superrun names start with an underscore. Regular run names cannot start with an underscore.

Strax builds data for a superrun by loading (and potentially building) each of the subruns, then
slicing and concatenating them as necessary. In addition superruns can be stored to disk as a
rechunked representation of its subruns. This currently only works for static lineages e.g. without
default-by-run_id settings. Stored superruns have the advantage that loading data is much faster
and different data_types of the same kind can be combined.

Superruns are useful to track common groupings of data. For example:

* 'Minimum bias' runs, consisting only of low-energy events, events passing some cuts, DM-candidates, PMT flashes, or other thing of interest. The low-level data of these is much smaller than that of all the full runs, and can be brought to a local analysis facility, enabling on-site low-level waveform watching.
* Grouping similar runs. For example, shifters might group good runs from a week of calibration data with some source under a single name, e.g. ``_kr_feb2019``.


Superruns can be built from other superruns. Thus, _sr1_v0.2 could be built from
_background_january, _background_february, etc.

Defining superruns and making data:
-----------------------------------
Use the `define_run` context method to define a new superrun. Currently it is only supported to
define superruns from a list of run_ids:


.. code-block:: python

    st.define_run('_awesome_superrun', ['123', '124'])


From a dictionary of time range tuples. The times must be 64-bit integer UTC timestamps since the unix epoch:

.. code-block:: python

        st.define_run('_awesome_superrun', {
            '123': [(start, stop), (start, stop), ...],
            '124': [(start, stop), (start, stop), ...],})

From a dataframe (or record array) with strax data:


.. code-block:: python

    st.define_run('_awesome_superrun', events_df)
    st.define_run('_awesome_superrun', events_df, from_run='123')

In this case, the run will be made of the time ranges that correspond exactly to ``events_df``. If ```events_df`` already has a ``run_id`` field (e.g. because it consists of data from multiple runs), you do not need to pass `from_run`, it will be read off from the data.

It is up to the storage frontend to process your request for defining a run. As a normal user, you
generally only have permissions to create a new run in the `DataDirectory` (local files) storage
frontend, where runs are recorded in json files.

Making superrun data is as easy as creating any other data. Once a superrun is defined we can make
for example event_info via:


.. code-block:: python

    st.make('_awesome_superrun', 'event_info)

For bookkeeping each stored superrun chunk contains information of its constituents in a field
called subruns e.g.:


.. code-block:: python

    {'0': {'end': 10, 'start': 0},
     '1': {'end': 30, 'start': 20},
     '2': {'end': 50, 'start': 40}}

Where the keys represent the subrun_ids and start/end the start and end of the corresponding
first/last chunk included in the superrun chunk. The same information can also be found in the
metadata of the individual chunks:

.. code-block:: python

    {'chunk_i': 0,
     'end': 50,
     'filename': 'records-j3nd2fjbiq-000000',
     'filesize': 2343,
     'first_endtime': 1,
     'first_time': 0,
     'last_endtime': 50,
     'last_time': 49,
     'n': 300,
     'nbytes': 77100,
     'run_id': '_superrun_test',
     'start': 0,
     'subruns': {'0': {'end': 10, 'start': 0},
                 '1': {'end': 30, 'start': 20},
                 '2': {'end': 50, 'start': 40}}}

After creating data we can load the superrun as we are used to and combine it with other data_types
of the same kind too.

To work more easily with superruns all chunks have also the properties `chunk.is_superun` as well as
`chunk.first_subrun` and `chunk.last_subrun`.

If you wish to make/store a superrun you have to specify the context option:


.. code-block:: python

    st.set_context_config({'write_superruns': True})


Superruns follow the same saving rules (SaveWhen.TARGET, SaveWhen.EXPLICIT or SaveWhen.ALWAYS) as regular runs. 

How superruns work
--------------------

As mentioned above, strax builds data for superruns by slicing data of the subruns. Thus, peaks
from a superrun come from the peaks of the subruns, which are built from their own records as usual.

Defaults for settings can be runid-dependent in strax, although this is not preferred any longer.
If an option specifies ``default_per_run=[(run, setting), (run2, setting2)]``, then runs in between
run and run2 will use setting, and runs after run2 ``setting2``. Superruns store a deterministic hash
of this ``default_per_run`` specification for tracking purposes.

You cannot currently go directly from the superrun's records to the superrun's peaks. This would be
tricky to implement, since (1) (2) even with the same settings, many plugins choose to do something
different depending on the run_id. For example, in straxen the gain model is specified by a file,
but which gains from the file are actually used is dependent on the runid.

Thus, superruns won't help build data faster, but they will speed up loading data after it has been
built. This is important, because strax' overhead for loading a run is larger than hax, due to its
version and option tracking (this is only true if per-run-default options are allowed).