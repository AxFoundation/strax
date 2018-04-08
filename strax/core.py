from itertools import chain
import threading
import logging

import numpy as np
import pandas as pd

import strax
export, __all__ = strax.exporter()

##
# Plugin registration
##

REGISTRY = dict()


@export
def register_plugin(plugin_class, provides=None):
    """Register plugin_class as provider for plugin_class.provides and
    other data types listed in provides.
    :param plugin_class: class inheriting from StraxPlugin
    :param provides: list of additional data types which this plugin provides.
    """
    if provides is None:
        provides = []
    global REGISTRY
    inst = plugin_class()
    for p in [inst.provides] + provides:
        REGISTRY[p] = inst
    return plugin_class


@export
def provider(data_name):
    """Return instance of plugin that provides data_name"""
    try:
        return REGISTRY[data_name]
    except KeyError:
        raise KeyError(f"No plugin registered that provides {data_name}")


@export
def data_info(data_name):
    """Return pandas DataFrame describing fields in data_name"""
    p = provider(data_name)
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
# Load system
##

class NotCachedException(Exception):
    pass


import os

from collections import namedtuple
CacheKey = namedtuple('CacheKey',
                      ('run_id', 'data_type', 'lineage'))


class FileCache:
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _dirname(key):
        return os.path.join(key.run_id, key.data_type)

    def get(self, key):
        """Return iterator factory over cached results,
        or raise NotCachedException if we have not cached the results yet
        """
        dirname = self._dirname(key)
        if os.path.exists(dirname):
            self.log.debug(f"{key} is in cache.")
            return strax.io_chunked.read_chunks(dirname)
        self.log.debug(f"{key} is NOT in cache.")
        raise NotCachedException

    def save(self, key, source):
        dirname = self._dirname(key)
        source = strax.chunk_arrays.fixed_size_chunks(source)
        strax.io_chunked.save_to_dir(source, dirname)


cache = FileCache()


@export
def get(run_id, target, save=None):
    log = logging.getLogger('get')
    if isinstance(save, str):
        save = [save]
    elif save is None:
        if provider(target).save_preference > strax.SavePreference.GRUDGINGLY:
            save = [target]
        else:
            save = []
    elif isinstance(save, tuple):
        save = list(save)

    mailboxes = dict()
    plugins_to_run = dict()
    threads = []
    keys = dict()    # Cache keys of the things we're building

    log.debug("Walking dependency graph")
    to_check = [target]
    while len(to_check):
        d = to_check.pop()
        if d in mailboxes:
            continue

        p = provider(d)
        mailboxes[d] = strax.OrderedMailbox(name=d + '_mailbox')
        keys[d] = CacheKey(run_id, d, p.lineage(run_id))
        try:
            cache_iterator = cache.get(keys[d])

        except NotCachedException:
            # We have to make this data
            plugins_to_run[d] = p
            if p.save_preference == strax.SavePreference.ALWAYS:
                save.append(d)
            # And we should check it's dependencies
            to_check.extend(p.depends_on)

        else:
            # Already in cache, read it from disk
            threads.append(threading.Thread(
                target=mailboxes[d].send_from,
                name='read:' + d,
                args=(cache_iterator,)))

    log.debug("Creating stream")

    for d in set(save):
        assert d in plugins_to_run
        if plugins_to_run[d].save_preference == strax.SavePreference.NEVER:
            raise ValueError("Plugin forbids saving data for {d}")
        threads.append(threading.Thread(
            target=cache.save,
            name='save:' + d,
            kwargs=dict(key=keys[d],
                        source=mailboxes[d].subscribe())))

    for d, p in plugins_to_run.items():
        threads.append(threading.Thread(
            target=mailboxes[d].send_from,
            name='build:' + d,
            args=(p.iter(iters={d: mailboxes[d].subscribe()
                                for d in p.depends_on}),)))

    final_generator = mailboxes[target].subscribe()

    # NB: start threads AFTER we've put in all subscriptions!
    log.debug("Starting threads")
    for t in threads:
        t.start()

    log.debug("Retrieving results")
    yield from final_generator

    log.debug("Closing threads")
    for t in threads:
        t.join(timeout=10)
        if t.is_alive():
            raise RuntimeError("Thread %s did not terminate!" % t.name)

# TODO: fix signatures

@export
def make(*args, **kwargs):
    for _ in get(*args, **kwargs):
        pass


@export
def get_array(*args, **kwargs):
    return np.concatenate(list(get(*args, **kwargs)))


@export
def get_df(*args, **kwargs):
    return pd.DataFrame.from_records(get_array(*args, **kwargs))
