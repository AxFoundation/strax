from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import inspect
import itertools

import numpy as np
import pandas as pd


ProcessorComponents = namedtuple('ProcessorComponents',
                                 ['plugins', 'loaders', 'savers'])

import strax
export, __all__ = strax.exporter()


@export
class Strax:
    """Streaming data processor"""

    def __init__(self, max_workers=None, storage=None):
        self.log = logging.getLogger('strax')
        if storage is None:
            storage = [strax.FileStorage()]
        self.storage = storage
        self._plugin_class_registry = dict()
        self._plugin_instance_cache = dict()
        self.executor = ThreadPoolExecutor(max_workers=max_workers,
                                           thread_name_prefix='pool_')
        # Hmm...
        for s in storage:
            s.executor = self.executor

        # Register placeholder for records
        # TODO: Hm, why exactly? And do I have to do this for all source
        # plugins?
        self.register(strax.RecordsPlaceholder)

    def register(self, plugin_class, provides=None):
        """Register plugin_class as provider for data types in provides.
        :param plugin_class: class inheriting from StraxPlugin
        :param provides: list of data types which this plugin provides.

        Plugins always register for the data type specified in the .provide
        class attribute. If such is not available, we will construct one from
        the class name (CamelCase -> snake_case)

        Returns plugin_class (so this can be used as a decorator)
        """
        if not hasattr(plugin_class, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = strax.camel_to_snake(plugin_class.__name__)
            plugin_class.provides = snake_name

        if provides is not None:
            provides += [plugin_class.provides]
        else:
            provides = [plugin_class.provides]

        for p in provides:
            self._plugin_class_registry[p] = plugin_class

        return plugin_class

    def register_all(self, module):
        """Register all plugins defined in module"""
        for x in dir(module):
            x = getattr(module, x)
            if type(x) != type(type):
                continue
            if issubclass(x, strax.Plugin):
                self.register(x)

    def data_info(self, data_name: str) -> pd.DataFrame:
        """Return pandas DataFrame describing fields in data_name"""
        p = self.get_components('SOME_RUN???', data_name).plugins[data_name]
        display_headers = ['Field name', 'Data type', 'Comment']
        result = []
        for name, dtype in strax.utils.unpack_dtype(p.dtype):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ''
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    ##
    # Processor creation
    ##

    def get_components(self,
                       run_id,
                       target: str,
                       save=tuple()):
        """Return components for setting up a processor

        :param run_id: run id to get
        :param target: data type to yield results for
        :param save: str or list of str of data types you would like to save
        to cache, if they occur in intermediate computations
        :param profile_to: filename to save profiling results to. If not
        specified, will not profile.
        """
        if isinstance(save, str):
            save = (save,)

        def get_plugin(d):
            nonlocal plugins

            if d not in self._plugin_class_registry:
                raise KeyError(f"No plugin class registered that provides {d}")
            p = self._plugin_class_registry[d]()

            p.log = logging.getLogger(p.__class__.__name__)
            if not hasattr(p, 'depends_on'):
                # Infer dependencies from self.compute's argument names
                process_params = inspect.signature(p.compute).parameters.keys()
                process_params = [p for p in process_params if p != 'kwargs']
                p.depends_on = tuple(process_params)

            plugins[d] = p

            p.deps = {d: get_plugin(d) for d in p.depends_on}
            print(d, p.depends_on, p.deps)

            p.lineage = {d: p.version(run_id)}
            for d in p.depends_on:
                p.lineage.update(p.deps[d].lineage)

            if not hasattr(p, 'data_kind'):
                if len(p.depends_on):
                    # Assume data kind is the same as the first dependency
                    p.data_kind = p.deps[p.depends_on[0]].data_kind
                else:
                    # No dependencies: assume provided data kind and
                    # data type are synonymous
                    p.data_kind = p.provides

            if not hasattr(p, 'dtype'):
                p.dtype = p.infer_dtype()
            p.dtype = np.dtype(p.dtype)
            return p

        plugins = defaultdict(get_plugin)
        # This works without the RHS too, but your IDE might not get it :-)
        plugins[target] = get_plugin(target)

        loaders = dict()
        savers = defaultdict(list)

        def check_cache(d):
            nonlocal plugins, loaders, savers
            p = plugins[d]
            key = strax.CacheKey(run_id, target, p.lineage)
            for sb_i, sb in enumerate(self.storage):
                try:
                    loaders[d] = sb.loader(key)
                    # Found it! No need to make it anymore:
                    del plugins[d]
                    return
                except strax.NotCachedException:
                    continue
            # Not in any cache. Maybe dependencies are?
            p = p
            for d in p.depends_on:
                check_cache(d)

            # Since we're making this, should we save it?
            if p.save_when == strax.SaveWhen.NEVER:
                if d in save:
                    raise ValueError("Plugin forbids saving of {d}")
                return
            elif p.save_when == strax.SaveWhen.TARGET:
                if d != target:
                    return
            elif p.save_when == strax.SaveWhen.EXPLICIT:
                if d not in save:
                    return
            else:
                assert p.save_when == strax.SaveWhen.ALWAYS

            for sb_i, sb in enumerate(self.storage):
                if not sb.provides(d):
                    continue
                s = sb.saver(key, plugins[d].metadata(run_id))
                savers[d].append(s)

        return ProcessorComponents(plugins=plugins,
                                   loaders=loaders,
                                   savers=savers)

    def simple_chain(self, run_id, source, target):
        chain = [target]
        plugins = dict()
        savers = dict()

        while chain[-1] != source:
            d = chain[-1]
            plugins[d] = p = self.provider(d)
            if len(p.depends_on) != 1:
                raise NotImplementedError("simple_chain does not support "
                                          "dependency branching")

            key = strax.CacheKey(run_id, d, p.lineage(run_id))
            for sb_i, sb in enumerate(self.storage):
                if not sb.provides(d):
                    continue
                savers.setdefault(d, [])
                savers[d].append(sb.saver(key,
                                          metadata=p.metadata(run_id)))

            chain.append(p.depends_on[0])
        chain = list(reversed(chain[:-1]))
        return SimpleChain(chain, plugins, savers)

    def get(self, **kwargs):
        """Compute target for run_id and iterate over results
        """
        components = self.get_components(**kwargs)
        return ThreadedMailboxProcessor(components,
                                        self.executor,
                                        kwargs['target'])

    def online(self, run_id: str, target: str,
               save=None, profile_to=None):
        """Return OnlineProcessor for computing target for run_id
        See .get for argument documentation. # TODO: fix
        """
        final_generator, mailboxes, plugins_to_run = self._prepare(
            run_id, target, save)

        source_plugins = {d: p for d, p in plugins_to_run.items()
                          if isinstance(p, strax.ReceiverPlugin)}
        if not len(source_plugins):
            raise ValueError("get_online is only needed if you have "
                             "online input sources.")

        def _spin():
            for _ in self._run(final_generator, mailboxes, profile_to):
                pass
        t = threading.Thread(target=_spin)
        t.start()
        return OnlineProcessor(t, source_plugins)

    # TODO: fix signatures
    def make(self, *args, **kwargs):
        for _ in self.get(*args, **kwargs):
            pass

    def get_array(self, *args, **kwargs):
        return np.concatenate(list(self.get(*args, **kwargs)))

    def get_df(self, *args, **kwargs):
        return pd.DataFrame.from_records(self.get_array(*args, **kwargs))



class ThreadedMailboxProcessor:

    def __init__(self, components, executor, target):
        self.log = logging.getLogger(self.__class__.__name__)

        plugins, loaders, savers = components
        self.mailboxes = dict()

        for d in itertools.chain.from_iterable(
                [c.keys() for c in components]):

            self.mailboxes[d] = m = strax.Mailbox(name=d + '_mailbox')

        for d in loaders:
            assert d not in plugins
            self.mailboxes[d].add_sender(loaders[d], name=f'load:{d}')

        for d in plugins:
            p = plugins[d]
            self.mailboxes[d].add_sender(p.iter(
                    iters={d: self.mailboxes[d].subscribe()
                           for d in p.depends_on},
                    executor=executor),
                name=f'build:{d}')

        for d in savers:
            for s_i, s in enumerate(savers[d]):
                self.mailboxes[d].add_reader(
                    s.save_from(self.mailboxes[d].subscribe()),
                    name=f'save_{s_i}:{d}')

        self.final_generator = self.mailboxes[target].subscribe()

    def __iter__(self):
        self.log.debug("Starting threads")
        for m in self.mailboxes.values():
            m.start()

        self.log.debug("Yielding results")
        try:
            yield from self.final_generator
        except strax.MailboxKilled:
            self.log.debug(f"Main thread received MailboxKilled")
            for m in self.mailboxes.values():
                self.log.debug(f"Killing {m}")
                m.kill(upstream=True,
                       reason="Strax terminating due to "
                              "downstream exception")

        self.log.debug("Closing threads")
        for m in self.mailboxes.values():
            m.cleanup()


class OnlineProcessor:
    """Online processor running in a background thread.
    Use send to submit in new values for inputs, close to terminate.
    TODO: how to crash-close? Do I need to?
    """
    def __init__(self, t, source_plugins):
        self.t = t
        self.source_plugins = source_plugins

    def send(self, data_type: str, chunk_i: int, data):
        """Send a new chunk (numbered chunk_i) of data_type"""
        self.source_plugins[data_type].send(chunk_i, data)

    def close(self, timeout=None):
        for s in self.source_plugins.values():
            s.close()
        # Now it needs some time to finish processing stuff that was waiting
        # in buffers
        self.t.join(timeout=timeout)
        if self.t.is_alive():
            raise RuntimeError("Online processing did not terminate on time!")


class SimpleChain:
    """A very basic processor that processes things on-demand
    without any background threading or rechunking.

    Suitable for high-performance multiprocessing.
    """
    def __init__(self, chain, plugins, savers):
        self.chain = chain
        self.plugins = plugins
        self.savers = savers

    def send(self, chunk_i, data):
        for d in self.chain:
            data = self.plugins[d].compute(data)
            for s in self.savers[d]:
                s.send(chunk_i=chunk_i, data=data)
        return data

    def close(self):
        for d in self.chain:
            for s in self.savers[d]:
                s.close()

    def crash_close(self):
        for d in self.chain:
            for s in self.savers[d]:
                s.crash_close()
