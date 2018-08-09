Storage
========

Overview
---------
Players in strax's storage system take on one of three roles:
  * ``StorageFrontend``: Find data locations, and communicate this to one or more ``StorageBackend`` instances;
  * ``StorageBackend``: load pieces of data, and create instances of ``Saver``;
  * ``Saver``: save pieces of data to a specific location.

As an example, a ``StorageFrontend`` could talk to a database that tracks which data is stored where.
A ``StorageBackend`` then retrieves data from local disks, while another might retrieve it remotely using SSH or other transfer systems.
The front-end decides which backend is appropriate for a given request. Finally, a ``Savers`` guides the process of writing a particular
piece of data to disk or databases (potentially from multiple cores), compressing and rechunking as needed.

To implement a new way of storing and/or tracking data, you must implement (subclass) all or some of these classes.
This means subclassing them and overriding a few specific methods
(called 'abstract methods' because they ``raise NotImplementedError`` if they are not overridden).

Keys
-----
In strax, a piece of data is identified by a *DataKey*, consisting of three components:
  * The run id
  * The data type
  * The complete *lineage* of the data. This includes, for the data type itself, and all types it depends on (and their dependencies, and so forth):
    * The plugin class name that produced the data;
    * The version string of the plugin;
    * The values of all configuration options the plugin took (whether they were explicitly specified or left as default).

When you ask for data using ``Context.get_xxx``, the context will produce a key like this, and pass it to the ``StorageFrontend``.
It then looks for a filename or database collection name that matches this key -- something a ``StorageBackend`` understands. which is therefore generically called a *backend key*.
The matching between DataKey and backend key can be done very strictly, or more loosely, depending on how the context is configured.
This way you can choose to be completely sure about what data you get, or be more flexible and load whatever is available.
TODO: ref context documentation.


Run-level metadata
-------------------
Metadata can be associated with a run, but no particular data type. The ``StorageFrontend`` must take care of saving and loading these.

Such run-level metadata can be crucial in providing run-dependent default setting for configuration options, for example, calibrated quantities necessary
for data processing (e.g. electron lifetime and PMT gains).
