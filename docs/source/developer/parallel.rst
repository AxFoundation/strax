Parallelization
================

Strax can process data at 50-100 raw-MB /sec single core, which is not enough for live online processing at high DAQ rates. We must thus parallelize at least some of the signal processing.

Not all plugins can be parallelized. For example, we cannot assign event numbers  (0, 1, 2, ...) in parallel if we want unique numbers that increment without gaps. We also cannot save to a single file in parallel.

Multithreading
---------------
To get parallelization, plugins can defer computations to a pool of **threads** or **processes**. If they do, they yield futures to the output mailbox instead of the actual results (numpy arrays). The mailbox awaits the futures and ensures each consumer gets the results in order.

A plugin indicates to strax it is paralellizable by setting its ``parallel`` attribute to True. This usually causes strax to outsource computations to a pool of threads. Every chunk will result in a call to the thread pool. This has little overhead, though the performance gain is limited by the global interpreter lock. If the computation is in pure python, there is no benefit; however, numpy and numba code can benefit significantly (until the pure-python overhead around it becomes the limiting factor, at high numbers of cores).

Loaders use multithreading by default, since their work is eminently parallelizable: they just load some data and decompress it (using low-level compressors that happily release the GIL). Savers that rechunk the data (e.g. to achieve more sysadmin-friendly filesizes) are not parallelizable. Savers that do not rechunk use multithreading just like loaders.


Multiprocessing
----------------

Strax can also use multiprocessing for parallelization. This is useful to free pure-python computations from the shackles of the GIL. Low-level plugins deal with a massive data flow, so parallelizing theircomputations in separate processes is very inefficient due to data transfer overhead. Thread parallelization works fine (since the algorithms are implemented in numba) until you reach ~10 cores, when the GIL becomes binding due to pure-python overhead. 

You can set the ``parallel`` attribute to ``process``, to suggest strax should use a process pool instead of a thread pool. This is often not a good idea: multiprocessing incurs overhead from (1) forking the strax process and (2) pickling and unpickling the results in the child and parent processes. Strax will still not use multiprocessing at all unless you:
  - Set the allow_multiprocess context option to True,
  - Set max_workers to a value higher than 1 in the get_xxx call.

During multiprocessing, computations of chunks from ``parallel='process'`` plugins will be outsourced to a process pool. Additionally, to avoid data transfer overhead, strax attempts to gather as many savers, dependencies, and savers of dependencies of a ``parallel='process'`` plugin to "inline" them: their computations are set to happen immedately after the main plugin's computation in the same process. This is achieved behind the scenes by replacing the plugin with a container-like plugin called ParallelSourcePlugin. Only parallelizable plugins and savers that do not rechunk will be inlined.

Since savers can become inlined, they should work even if they are forked. That implies they cannot keep state, and must store metadata for each chunk in their backend as it arrives. For example, the FileStore backend produces a json file with metadata for each chunk. When the saver is closed, all the json files are read in and concatenated. A saver that can't do this should set `allow_fork = False`. 


Multi-run parallelization: a different problem
------------------------------------------------

Paralellizing quick (re)processing of many runs is a different problem altogether. It is easier in one way: since runs are assumed to be independent, we can simply process each run on a single core, and use our multiple cores to process multiple runs. However, it is harder in another: the large volume of desired result data may exceed available RAM. We can use Dask dataframes for this. Probably we can just copy/reuse the code in hax.
