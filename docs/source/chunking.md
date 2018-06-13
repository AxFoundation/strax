# Developer notes: processing

This describes how strax's processing framework works under the hood, and explains some implementation choices. It's meant for people who want to do core development on strax; users or even plugin developers should not need it.

## Chunking and synchronization
Plugins receive and produce data in *chunks*. Plugins can chunk their output as they wish, but must not assume anything about how their inputs are chunked. Two inputs, even of the same data kind, will not always arrive chunked in the same way. Plugins therefore *synchronize* their inputs -- transform to equal-size merged chunks -- before computations. This is done under the hood in the plugin base class.

Savers are free to chunk their data as they like; for example, in files of convenient sizes. [TODO; exception for parallel blah...]

### Example: a plugin that must rechunk
Imagine a plugin that computes the time between a peak and the next peak. This can't compute time for the final peak in a chunk.
  * The first output chunk will thus be one item smaller. However, it will save the time of that last peak for later use.
  * Once the next input chunk arrives, it computes the time difference between the saved peak and the first peak of the new input chunk. This is the first result of the second output chunk. The second output chunk has the same size as the input chunk, but it will contain different peaks.
  * Finally, the last chunk will be one item larger.


### Synchronization of a single data kind 
Data types of the same *data kind* (e.g. 'peak_basics' and 'peak_classification') must have the same number of total items per run, and each item refers to a property of the same object (the underlying peak). Synchronizing inputs of the same kind is thus easy: simply regroup the inputs in same-length sequences. 

### What if I want to add or remove items? 
If your plugin removes or adds some items, it must output a different *data kind*. For example, during the initial data reduction steps, we remove items from 'raw_records' to make a new data kind 'records'. Here we change data kind, even though the fields in the output data type are identical to the fields in the input data type. 

### Synchronization of multiple kinds
Plugins that have multiple data kinds as input synchronize them on *temporal containment*. For concreteness, suppose a plugin takes data of kinds 'events' and 'peaks'. It must designate one kind -- say events -- as the base kind, which decides synchronization. With each chunk of events, we take all peaks that are fully contained in (start and end inside) one of the events. Peaks in between events are filtered out. 

In the future, more advanced kinds of synchronization would be useful.


## Pipeline

### Push versus pull

In python, pipeline components can offer two semantics. In **pull-semantics**, usually implemented with generators, somebody calls `next` to pull output, and `StopIteration` signals nothing more is coming. In **push-semantics**, usually implemented with coroutines, input is pushed in with a `send` method.  If cleanup is required, a `close` method must be invoked. These can be chained together to make pipelines. Either can also be implemented with custom classes instead of standard python generators/coroutines.

Strax primarily uses pull-semantics:
  * Loaders are plain iterators;
  * Plugins iterate over inputs, and expect their results to be iterated over;
  * Savers use both semantics. Usually they iterate over their input. However, during multiprocessing, savers have their inputs sent into them, and must be explicitly closed.


### Mailboxes
Strax could not be built by just chaining iterators or coroutines:
  * Pipelines can have multiple inputs and outputs, which generally come at different speeds; we cannot simply push on or pull from one endpoint.
  * For parallellization, we must run the same computation on several chunks at a time, then gather the results.

The *mailbox* class provides the additional machinery that handles this. During processing, each data type has a mailbox. 
A data type's mailbox iterates over the results of the plugin or loader that produces it. It also provides an iterator to each plugin that needs it as an input. 

The input iterators given to the plugins must be somewhat magical. If we call `next`, but the input is not yet available, we must pause (and do something else) until it is.
To enable this suspending, strax runs each plugin in a separate thread. (We could use a framework like `asyncio` instead if we wanted to run our own scheduler, now we just use the OS' scheduler.)

The threads in strax are thus motivated by [concurrency](https://en.wikipedia.org/wiki/Concurrency_(computer_science)), not parallelism.
As a bonus, they do allow different plugins to run simultaneously.
The benefit is limited by python's global interpreter lock, 
but this does not affect IO or computations in numpy and numba.

### Exception propgagation

...


## Parallelization

Strax can process data at 50-100 raw-MB /sec single core, which is not enough for live online processing at high DAQ rates. We must thus parallelize at least some of the signal processing. 

This might be impossible. For example, we cannot assigning event numbers  (0, 1, 2, ...) in parallel if we want unique numbers that increment without gaps. We also cannot save to a single file in parallel. 


### Implementation

To get parallelization, plugins can defer computations to a pool of **threads** or **processes**. If they do, they yield futures to the output mailbox instead of the actual results (numpy arrays). The mailbox awaits the futures and ensures each consumer gets the results in order.

Each plugin chooses whether it wants to use a thread- or process pool for parallelization, or cannot be parallelized at all.
  * A *thread pool* has little overhead, but the performance gain is limited by the global interpreter lock. If the computation is in pure python, there is no benefit at all. Numba code reaps benefits until the pure-python overhead around it becomes the limiting factor (at high numbers of cores).  
  * A *process pool* incurs overhead from (1) forking the strax process and (2) pickling and unpickling the results in the child and parent processes. We hope to remove overhead (2) using a shared memory system. For example, [zeroMQ](http://zeromq.org/) allows interprocess communication without copying, and [works for numpy arrays specifically](http://pyzmq.readthedocs.io/en/latest/serialization.html). 

Loaders always use a threadpool for paralellization. Savers used to do the same, but currently this is not supported, and saving is not usually parallelized. Savers that are optimized to post-compute operations (see below) are the exception.

### Chain optimization

...

### Multi-run parallelization: a different problem

Paralellizing quick (re)processing of many runs is a different problem altogether. Since runs are assumed to be independent, we can process each run in parallel on a single core. However, here we face a large volume of result data, which may exceed available RAM. We can use Dask dataframes for this. Probably we can just copy/reuse the code in hax for this.
