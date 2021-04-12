Recompressing & moving data
===========================
In order to recompress data with another compression algorithm the
:py:func:`context.copy_to_frontend` function can be used.
The function works on a per run_id, per datatype basis. In the example
below, peaks data is copied to a second frontend.


.. code-block:: python

    import strax
    import os
    # Naturally, these plugins (Records and Peaks) only serve as examples
    # and are best replaced by a fully constructed context
    from strax.testutils import Records, Peaks, run_id

    # Initialize context (st):
    st = strax.Context(register=[Records, Peaks])

    # Initialize frontends
    storage_frontend_A = strax.DataDirectory('./folder_A')
    storage_frontend_B = strax.DataDirectory('./folder_B',
                                          readonly=True)
    st.storage = [storage_frontend_A,
                  storage_frontend_B]

    # In this example, we will only consider records
    target = "records"

    print(f'Are records stored?\n{st.is_stored(run_id, target)}')

    # Make the data (stores to every frontend available)
    st.get_array(run_id, 'records')

    for sf in st.storage:
        print(f'{target} stored in\n\t{sf}?\n\t{st._is_stored_in_sf(run_id, target, sf)}')

Which prints:

.. code-block:: rst

    Are records stored?
    False
    records stored in
        strax.storage.files.DataDirectory, path: ./folder_A?
        True
    records stored in
        strax.storage.files.DataDirectory, readonly: True, path: ./folder_B?
        False

Copy
____
In the example above the `storage_frontend_B` was readonly, therefore,
when creating records, no is data stored there.
Below, we will copy the data from `storage_frontend_A` to
`storage_frontend_B`.

.. code-block:: python

    # First set the storage_frontend_B for readonly=False such that we can copy
    # data there
    storage_frontend_B.readonly = False

    # In the st.storage-list, storage_frontend_B is index 1
    index_frontend_B = 1
    st.copy_to_frontend(run_id, target,
                        target_frontend_id=index_frontend_B)

    for sf in [storage_frontend_A,  storage_frontend_B]:
        print(f'{target} stored in\n\t{sf}?\n\t{st._is_stored_in_sf(run_id, target, sf)}')


Which prints the following (so we can see that the copy to `folder_B`
was successful.

.. code-block:: rst

    records stored in
        strax.storage.files.DataDirectory, path: ./folder_A?
        True
    records stored in
        strax.storage.files.DataDirectory, path: ./folder_B?
        True

Copy and recompress
___________________
Now, with a third storage frontend, we will recompress the data to
reduce the size on disk.

.. code-block:: python

    # Recompression with a different compressor
    # See strax.io.COMPRESSORS for more compressors
    target_compressor = 'bz2'

    # Add the extra storage frontend
    index_frontend_C = 2
    storage_frontend_C = strax.DataDirectory('./folder_C')
    st.storage.append(storage_frontend_C)

    # Copy and recompress
    st.copy_to_frontend(run_id, target,
                        target_frontend_id=index_frontend_C,
                        target_compressor=target_compressor)

    for sf in st.storage:
        first_cunk = os.path.join(sf.path,
                                 '0-records-sqcyyhsfpv',
                                 'records-sqcyyhsfpv-000000')
        print(f'In {sf.path}, the first chunk is {os.path.getsize(first_cunk)} kB')

Which outputs:

.. code-block:: rst

    In ./folder_A, the first chunk is 275 kB
    In ./folder_B, the first chunk is 275 kB
    In ./folder_C, the first chunk is 65 kB

From the output we can see that the size of the first chunk of
folder_C, the data much smaller than in folder_A/folder_B. This comes
from the fact that `bz2` compresses the data much more than the default
compressor `blosc`.

How does this work?
__________________
Strax knows from the metadata stored with the data with witch
compressor the data was written. It is possible to use a different
compressor when re-writing the data to disk (as done for `strax` knows
from the metadata stored with the data with witch compressor the data
was written. It is possible to use a different compressor when
re-writing the data to disk (as done folder_C in the example above).

As such, for further use, it does not matter if the data is coming from
either of folders folder_A-folder_C as the metadata will tell strax
which compressor to use. Different compressors may have different
performance for loading/writing data.