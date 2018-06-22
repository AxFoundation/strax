Chunking and synchronization
============================
Plugins receive and produce data in *chunks*. Plugins can chunk their output as they wish, but must not assume anything about how their inputs are chunked. Two inputs, even of the same data kind, will not always arrive chunked in the same way. Plugins therefore *synchronize* their inputs -- transform to equal-size merged chunks -- before computations. This is done under the hood in the plugin base class.

Savers are free to chunk their data as they like; for example, in files of convenient sizes. For very low-level plugins this is not recommended, as it has consequences for parallelization (see below).

Example: a plugin that must rechunk
-------------------------------------
Imagine a plugin that computes the time between a peak and the next peak. This can't compute the result for the final peak in a chunk.

* The first output chunk will thus be one item smaller. However, it will save the time of that last peak for later use.
* Once the next input chunk arrives, it computes the time difference between the saved peak and the first peak of the new input chunk. This is the first result of the second output chunk. The second output chunk has the same size as the input chunk, but it will contain different peaks.
* Finally, the last chunk will be one item larger.

Synchronization of one data kind
--------------------------------------
Data types of the same *data kind* (e.g. 'peak_basics' and 'peak_classification') must have the same number of total items per run, and each item refers to a property of the same object (the underlying peak). Synchronizing inputs of the same kind is thus easy: simply regroup the inputs in same-length sequences.

If your plugin removes or adds some items, it must thus output a different *data kind*. For example, during the initial data reduction steps, we remove items from 'raw_records' to make a new data kind 'records'. Here we change data kind, even though the fields in the output data type are identical to the fields in the input data type.

Synchronization of multiple kinds
-----------------------------------
Plugins that have multiple data kinds as input synchronize them on *temporal containment*. For concreteness, suppose a plugin takes data of kinds 'events' and 'peaks'. It must designate one kind -- say events -- as the base kind, which decides synchronization. With each chunk of events, we take all peaks that are fully contained in (start and end inside) one of the events. Peaks in between events are filtered out.

In the future, more advanced kinds of synchronization would be useful.
