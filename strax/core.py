import strax
import pandas as pd

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


class FakeCache:
    def get(self, key):
        dirn = os.path.join(key.run_id, key.data_type)
        if os.path.exists(dirn):
            print(f"{key} is in cache!")
            return lambda: strax.io_chunked.read_chunks(dirn)
        print(f"{key} not in cache.")
        raise NotCachedException


cache = FakeCache()


@export
def get(run_id, target):
    # Plugins and mailboxes for just the things we have to load
    plugins = dict()
    mailboxes = dict()
    # Iterator-factories for all things we need (cached or not)
    sources = dict()

    stack = [target]
    while len(stack):
        d = stack.pop()
        if d in sources:
            continue
        p = provider(d)

        key = CacheKey(run_id, d, p.lineage(run_id))
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
            # And we should check it's dependencies
            stack.extend(p.depends_on)

    threads = {}
    import threading
    for d, p in plugins.items():
        iters = {d: sources[d]() for d in p.depends_on}
        threads[d] = threading.Thread(target=p.iter,
                                      name=d,
                                      args=(iters, mailboxes[d]))
        threads[d].start()

    yield from mailboxes[target].subscribe()
