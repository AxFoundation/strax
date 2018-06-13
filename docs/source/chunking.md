## Chunking
Plugins receive and produce data in *chunks*. Plugins can chunk their output however they wish, and in return assume anything about how their inputs are chunked. For example, two inputs, even of the same data kind, will generally not be chunked in the same way. Plugins thus *synchronize* their inputs first; this is done under the hood in the plugin base class.

Savers are also free to chunk their data as they like, for example, in files of convenient sizes.

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
Plugins that have multiple data kinds as input synchronize them on *temporal containment*. For concreteness, take the data kinds 'events' and 'peaks'. A plugin designates one kind -- say events -- as its base input kind, which decides synchronization. With each chunk of events, we then take all peaks that are fully contained in (start and end inside) one of the events. Peaks in between events are filtered out. In the future, more advanced kinds of synchronization would be useful.


## Parallelization

Strax can process data at 50-100 raw-MB /sec single core, which is not enough for live online processing at high DAQ rates. We must thus parallelize at least some of the signal processing plugins. 

This might be impossible. For example, we cannot assigning event numbers  (0, 1, 2, ...) in parallel if we want unique numbers that increment without gaps. We also cannot save to a single file in parallel. 

Paralellizing quick (re)processing of many runs is a different challenge. Since runs are assumed to be independent, we can multiprocess (without shared data) each run separately. The challenge here is dealing with the large volume of result data, which may exceed available RAM.

## Implementation choices

Loaders produce data, savers consume them, plugins do both. How to implement them?

Besides a variety of frameworks (asyncio, streamz, curio, dask, ...) there are two basic pipeline components:
  * **Pull-semantics: iterators**. Somebody must call `next` to get output. `StopIteration` signals nothing more coming.
  * **Push-semantics: coroutines**. Somebody must send in input using `send`. If cleanup required, must send `close()` as well. These can be beefed-up generators (using `generator.send()` and `yield from`) or use the `async def` and `await` features of python 3.5+.  
  
Both can also be custom classes with `send` or `next` methods. Pax plugins were custom classes with push-semantics, except for input plugins, which supported both modes (pushing in event numbers for random access, or pulling to iterate over all events).

Clearly loaders can be generators, and savers coroutines.
