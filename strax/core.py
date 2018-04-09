from copy import copy
import threading
import logging
import inspect

import numpy as np
import pandas as pd

import strax
__all__ = 'Strax', 'register_default'


class Strax:
    """Streaming data processor"""

    # Yes, that's a class-level mutable, so register_default works
    _plugin_class_registry = dict()

    def __init__(self, storage_backends=None):
        self.log = logging.getLogger('strax')
        if storage_backends is None:
            storage_backends = [strax.FileStorage()]
        self.storage_backends = storage_backends
        self._plugin_class_registry = copy(self._plugin_class_registry)
        self._plugin_instance_cache = dict()

    def register(self, plugin_class, provides=None):
        """Register plugin_class as provider for data types in provides.
        :param plugin_class: class inheriting from StraxPlugin
        :param provides: list of data types which this plugin provides.

        Plugins always register for the data type specified in the .provide
        class attribute. If such is not available, we will construct one from
        the class name (CamelCase -> snake_case)

        Returns plugin_class (so this can be used as a decorator)
        """
        if provides is None:
            provides = []

        if not hasattr(plugin_class, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = strax.camel_to_snake(plugin_class.__name__)
            plugin_class.provides = snake_name

        for p in [plugin_class.provides] + list(provides):
            self._plugin_class_registry[p] = plugin_class

        return plugin_class

    def provider(self, data_name):
        """Return instance of plugin that provides data_name

        This is the only way plugins should be instantiated.
        """
        if data_name not in self._plugin_class_registry:
            raise KeyError(f"No plugin registered that provides {data_name}")
        if data_name in self._plugin_instance_cache:
            return self._plugin_instance_cache[data_name]

        p = self._plugin_class_registry[data_name]()
        p.log = logging.getLogger(p.__class__.__name__)

        if not hasattr(p, 'depends_on'):
            # Infer dependencies from self.compute's argument names
            process_params = inspect.signature(p.compute).parameters.keys()
            process_params = [p for p in process_params if p != 'kwargs']
            p.depends_on = tuple(process_params)

        if not hasattr(p, 'data_kind'):
            # Assume data kind is the same as the first dependency
            p.data_kind = self.provider(p.depends_on[0]).data_kind

        p.dependency_kinds = {d: self.provider(d).data_kind
                              for d in p.depends_on}
        p.dependency_dtypes = {d: self.provider(d).dtype
                               for d in p.depends_on}

        if not hasattr(p, 'dtype'):
            p.dtype = p.infer_dtype()
        p.dtype = np.dtype(p.dtype)

        p.startup()

        return p

    def data_info(self, data_name):
        """Return pandas DataFrame describing fields in data_name"""
        p = self.provider(data_name)
        display_headers = ['Field name', 'Data type', 'Comment']
        result = []
        for name, dtype in strax.utils.unpack_dtype(p.dtype):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ''
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    def get(self, run_id, target, save=None):
        """Compute target for run_id and iterate over results
        :param run_id: run id to get
        :param target: data type to yield results for
        :param save: str or list of str of data types you would like to save
        to cache, if they occur in intermediate computations
        """
        if isinstance(save, str):
            save = [save]
        elif save is None:
            if (self.provider(target).save_preference
                    > strax.SavePreference.GRUDGINGLY):
                save = [target]
            else:
                save = []
        elif isinstance(save, tuple):
            save = list(save)

        mailboxes = dict()
        plugins_to_run = dict()
        threads = []
        keys = dict()    # Cache keys of the things we're building

        self.log.debug("Creating saver/loader threads and mailboxes")
        to_check = [target]
        while len(to_check):
            d = to_check.pop()
            if d in mailboxes:
                continue

            p = self.provider(d)
            if d in save and p.save_preference == strax.SavePreference.NEVER:
                raise ValueError("Plugin forbids saving of {d}")
            mailboxes[d] = strax.OrderedMailbox(name=d + '_mailbox')
            keys[d] = strax.CacheKey(run_id, d, p.lineage(run_id))

            for sb in self.storage_backends:
                try:
                    cache_iterator = sb.get(keys[d])
                except strax.NotCachedException:
                    continue
                else:
                    # Create reader thread
                    threads.append(threading.Thread(
                        target=mailboxes[d].send_from,
                        name='read:' + d,
                        args=(cache_iterator,)))
                    break

            else:
                # We have to make this data, and possibly its dependencies
                plugins_to_run[d] = p
                to_check.extend(p.depends_on)

                # Create saver threads
                if (d in save
                        or p.save_preference == strax.SavePreference.ALWAYS):
                    for sb_i, sb in enumerate(self.storage_backends):
                        if not sb.provides(d):
                            continue
                        threads.append(threading.Thread(
                            target=sb.save,
                            name=f'save_{sb_i}:{d}',
                            kwargs=dict(key=keys[d],
                                        source=mailboxes[d].subscribe(),
                                        metadata=p.metadata(run_id))))

        self.log.debug("Creating builder threads")
        for d, p in plugins_to_run.items():
            threads.append(threading.Thread(
                target=mailboxes[d].send_from,
                name='build:' + d,
                args=(p.iter(iters={d: mailboxes[d].subscribe()
                                    for d in p.depends_on}),)))

        final_generator = mailboxes[target].subscribe()

        # NB: do this AFTER we've put in all subscriptions!
        self.log.debug("Starting threads")
        for t in threads:
            t.start()

        self.log.debug("Yielding results")
        yield from final_generator

        self.log.debug("Closing threads")
        for t in threads:
            t.join(timeout=10)
            if t.is_alive():
                raise RuntimeError("Thread %s did not terminate!" % t.name)

    # TODO: fix signatures
    def make(self, *args, **kwargs):
        for _ in self.get(*args, **kwargs):
            pass

    def get_array(self, *args, **kwargs):
        return np.concatenate(list(self.get(*args, **kwargs)))

    def get_df(self, *args, **kwargs):
        return pd.DataFrame.from_records(self.get_array(*args, **kwargs))


def register_default(plugin_class, provides=None):
    """Register plugin_class with all Strax processors created afterwards.
    Does not affect Strax'es already initialized
    """
    return Strax.register(Strax, plugin_class, provides)
