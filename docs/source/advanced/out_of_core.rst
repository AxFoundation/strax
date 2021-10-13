Out of core computation
=======================

Overview and motivation
------------------------
Many times analyses (performing some computation not implemented by a plugin or plotting) require loading more data than can fit into a memory,
these type of tasks are commonly reffered to as out-of-core computations.
Out-of-core algorithms usually involve a few repeating steps:
1) chunk the dataset into managable sizes
2) load the data chunk by chunk
3) perform some computation on each chunk
4) saving a summary of the results for each chunk
5) perform some combination of the results into a final result. 

While it is of course possible to implement these operations yourself, it can be tedious and repetative and the code becomes very rigid to the specific calculations being performed.
A better approach is to use abstractions of commonly performed operations that use out-of-core algorithms under the hood to get the same result as if the operations were performed on the entire dataset.
Code written using these abstractions can then run both on in-memory datasets as well as out-of-core datasets alike.
More importantly the implmentations of these algorithms can be written once and packaged to then be used by all. 

Data chunking
-------------
The zarr package provides an abstraction of the data-access api of numpy arrays for chunked and compressed data stored in memory or disk.
zarr provides an array abstraction with identical behavior to a numpy array when accessing data but where the underlyign data is actually a collection of compressed (optional) chunks.
the strax context provides a convenience method for loading data directly into zarr arrays. 

.. code-block:: python

    import strax

    context = strax.Context(**CONTEXT_KWARGS)

    # you can pass the same arguments you pass to context.get_array()
    zgrp = context.get_zarr(RUN_IDs, DATA_TYPES, **GET_ARRAY_KWARGS)

    # the zarr group contains multiple arrays, one for each data type
    z = zgrp.data_type 

    # individual arrays are also accessible via the __getitem__ interface
    z = zgrp['data_type']

    # numpy-like data access, abstracting away the underlying
    # data reading which may include readin multiple chunks from disk/memory
    # and decompression then concatenation to return an in memory numpy array 
    z[:100]


Data processing
---------------
The dask package provides abstractions for most of the numpy and pandas apis.
The dask.Array and dask.DataFrame objects implement their respective apis 
using fully distributed algorithms, only loading a fraction of the total data into memory
at any given moment for a given computing partition (thread/process/HPC-job).

.. code-block:: python

    import dask.array as da
    
    # easily convert to dask.Array abstraction for processing
    darr = da.from_zarr(z) 

    # its recommended to rechunk to sizes more appropriate for processing
    # see dask documentation for details
    darr.rechunk(CHUNK_SIZE)

    # you can also convert the dask.Array abstraction
    # to a dask.DataFrame abstraction if you need the pandas api
    ddf = darr.to_dask_dataframe()