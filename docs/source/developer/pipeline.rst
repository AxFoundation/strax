Pipeline
=========

This describes how strax chains computations from multiple plugins together in a pipeline.

In python, pipeline components can offer two semantics. In **pull-semantics**, usually implemented with generators, somebody calls `next` to pull output, and `StopIteration` signals nothing more is coming. In **push-semantics**, usually implemented with coroutines, input is pushed in with a `send` method.  If cleanup is required, a `close` method must be invoked. These can be chained together to make pipelines. Either can also be implemented with custom classes instead of standard python generators/coroutines.

Strax primarily uses pull-semantics:
  * Loaders are plain iterators;
  * Plugins iterate over inputs, and expect their results to be iterated over;
  * Savers use both semantics. Usually they iterate over their input. However, during multiprocessing, savers have their inputs sent into them, and must be explicitly closed.

Mailboxes
----------
Strax could not be built by just chaining iterators or coroutines.
  * Pipelines can have multiple inputs and outputs, which generally come at different speeds; we cannot simply push on or pull from one endpoint.
  * For parallellization, we must run the same computation on several chunks at a time, then gather the results.

The *mailbox* class provides the additional machinery that handles this. During processing, each data type has a mailbox.
A data type's mailbox iterates over the results of the plugin or loader that produces it. It also provides an iterator to each plugin that needs it as an input.

The input iterators given to the plugins must be somewhat magical. If we call `next`, but the input is not yet available, we must pause (and do something else) until it is.
To enable this suspending, strax runs each plugin in a separate thread. (We could use a framework like `asyncio` instead if we wanted to run our own scheduler, now we just use the OS's scheduler.)

The threads in strax are thus motivated by `concurrency <https://en.wikipedia.org/wiki/Concurrency_(computer_science)>`_, not parallelism. As a bonus, they do allow different plugins to run simultaneously. The benefit is limited by python's global interpreter lock, but this does not affect IO or computations in numpy and numba.



Exception propgagation
------------------------

TODO: document MailboxKilled etc.
