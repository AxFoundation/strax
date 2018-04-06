import threading

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
        self.log = strax.setup_logger('cache')

    @staticmethod
    def _dirname(key):
        return os.path.join(key.run_id, key.data_type)

    def get(self, key):
        dirname = self._dirname(key)
        if os.path.exists(dirname):
            # TODO: we currently read from cache multiple times
            # if a cached dependency is required multiple times.
            # Better to read once and fill a mailbox?
            self.log.debug(f"{key} is in cache.")
            return lambda: strax.io_chunked.read_chunks(dirname)
        self.log.debug(f"{key} is NOT in cache.")
        raise NotCachedException

    def save(self, key, source):
        dirname = os.path.join(key.run_id, key.data_type)
        source = strax.chunk_arrays.fixed_size_chunks(source)
        strax.io_chunked.save_to_dir(source, dirname)


cache = FileCache()


@export
def get(run_id, target, save=None):
    if isinstance(save, str):
        save = [save]
    elif save is None:
        if provider(target).save_preference > strax.SavePreference.GRUDGINGLY:
            save = [target]
        else:
            save = []
    elif isinstance(save, tuple):
        save = list(save)

    # For just the things we have to load:
    plugins = dict()
    mailboxes = dict()
    keys = dict()           # Cache keys

    # For all things we need (cached or not)
    sources = dict()        # Iterator factories

    stack = [target]
    while len(stack):
        d = stack.pop()
        if d in sources:
            continue
        p = provider(d)

        key = keys[d] = CacheKey(run_id, d, p.lineage(run_id))
        try:
            sources[d] = cache.get(key)
            if d == target:
                # We actually have the main target in cache!
                # No need to go any further
                yield from sources[d]()
                return

        except NotCachedException:
            # We'll make this data
            plugins[d] = p
            mailboxes[d] = strax.OrderedMailbox(name=d + '_mailbox')
            sources[d] = mailboxes[d].subscribe
            if p.save_preference == strax.SavePreference.ALWAYS:
                save.append(d)
            # And we should check it's dependencies
            stack.extend(p.depends_on)

    saver_threads = {}
    for d in set(save):
        assert d in plugins
        if plugins[d].save_preference == strax.SavePreference.NEVER:
            raise ValueError("Plugin forbids saving data for {d}")
        saver_threads[d] = t = threading.Thread(
            target=cache.save,
            name=d + '_saver',
            args=(keys[d], sources[d]()))
        t.start()

    plugin_threads = {}
    for d, p in plugins.items():
        iters = {d: sources[d]()
                 for d in p.depends_on}
        plugin_threads[d] = t = threading.Thread(
            target=p.iter,
            name=d,
            args=(iters, mailboxes[d]))
        t.start()

    yield from mailboxes[target].subscribe()


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
