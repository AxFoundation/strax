Parallelization
================

Strax can process data at 50-100 raw-MB /sec single core, which is not enough for live online processing at high DAQ rates. We must thus parallelize at least some of the signal processing.

Not all plugins can be parallelized. For example, we cannot assign event numbers  (0, 1, 2, ...) in parallel if we want unique numbers that increment without gaps. We also cannot save to a single file in parallel.

Indicating a parallelizable plugin
-----------------------------------
To get parallelization, plugins can defer computations to a pool of **threads** or **processes**. If they do, they yield futures to the output mailbox instead of the actual results (numpy arrays). The mailbox awaits the futures and ensures each consumer gets the results in order.

A plugin indicates to strax it is paralellizable by setting its `parallel` attribute to True. This usually causes strax to outsource computations to a pool of threads. Every chunk will result in a call to the thread pool. This has little overhead, but the performance gain is limited by the global interpreter lock. If the computation is in pure python, there is no benefit at all. Numba code reaps benefits until the pure-python overhead around it becomes the limiting factor (at high numbers of cores).

You can also set the `parallel` attribute to `process`, to indicate you would rather strax uses a process pool instead of a thread pool. This frees even a pure-python computation from the global interpreter lock, but incurs overhead from (1) forking the strax process and (2) pickling and unpickling the results in the child and parent processes.

Saver and loaders are currently not parallelized at all, except for savers that are optimized away by `ParallelSourcePlugin`s (see below).

ParallelSourcePlugin
---------------------

Low-level plugins deal with a massive data flow, so parallelizing their computations in separate processes is very inefficient due to data transfer overhead. Thread parallelization works fine (since the algorithms are implemented in numba) until you reach ~10 cores, when the global interpreter lock becomes binding due to pure-python overhead. We thus need a third parallelization mode: the `ParallelSourcePlugin`. This is how the `DAQReader` plugin is implemented.

Computations of chunks from a `ParallelSourcePlugin` will be outsourced to a process pool, just like `parallel='process'`. However, during setup, it will also gather as many of its dependencies and their savers as 'subsidiaries'. Their computations are then "inlined", that is, happen immedately after the main computation in the same process.

A plugin becomes inlined as a subsidiary if two conditions are met:
  * It can be paralellized (`parallel=True` or `parallel=process`)
  * It has only one dependency, which must be the `ParallelSourcePlugin` itself or another plugin that is inlined.

A saver becomes inlined if two conditions are met:
  * It belongs to a plugin that is inlined;
  * It does not rechunk the data (e.g. split it into convenient file sizes).

The resulting combination of outputs are first collected in a single mailbox (named after the `ParallelSourcePlugin`), which is unique in receiving dictionaries of arrays instead of just arrays. A tiny `send_outputs` function then distributes these appropriately to other mailboxes in the pipeline.

Since savers can become inlined, they must work even if they are forked. That implies they cannot keep state, and must store metadata for each chunk in their backend as it arrives. For example, the FileStore backend produces a json file with metadata for each chunk. When the saver is closed, all the json files are read in and concatenated.


Multi-run parallelization: a different problem
------------------------------------------------

Paralellizing quick (re)processing of many runs is a different problem altogether. It is easier in one way: since runs are assumed to be independent, we can simply process each run on a single core, and use our multiple cores to process multiple runs. However, it is harder in another: the large volume of desired result data may exceed available RAM. We can use Dask dataframes for this. Probably we can just copy/reuse the code in hax.
