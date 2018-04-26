import collections
from concurrent.futures import ThreadPoolExecutor
import logging
import inspect
import itertools
import typing as ty

import numpy as np
import pandas as pd

import strax
export, __all__ = strax.exporter()


@export
class Strax:
    """Streaming analysis for XENON (or eXperiments?)

    Specify how data should be processed, then start processing.
    """

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
        p = self._get_plugins((data_name,))[data_name]
        display_headers = ['Field name', 'Data type', 'Comment']
        result = []
        for name, dtype in strax.utils.unpack_dtype(p.dtype):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ''
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    def _get_plugins(self,
                     targets: ty.Tuple[str] = tuple(),
                     run_id: str = 'UNKNOWN'
                     ) -> ty.Dict[str, strax.Plugin]:
        """Return dictionary of plugins necessary to compute targets
        from scratch.
        """
        # Initialize plugins for the entire computation graph
        # (most likely far further down than we need)
        # to get lineages and dependency info.
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

        plugins = collections.defaultdict(get_plugin)
        for t in targets:
            # This works without the RHS too, but your IDE might not get it :-)
            plugins[t] = get_plugin(t)

        return plugins

    def get_components(self, run_id: str,
                       targets=tuple(), save=tuple(), sources=tuple()
                       ) -> strax.ProcessorComponents:
        """Return components for setting up a processor

        :param run_id: run id to get
        :param targets: data type to yield results for
        :param save: str or list of str of data types you would like to save
        to cache, if they occur in intermediate computations
        :param sources: str of list of str of data types you will feed the
        processor via .send.
        """
        def to_str_tuple(x) -> ty.Tuple[str]:
            if isinstance(x, str):
                return x,
            elif isinstance(x, list):
                return tuple(x)
            return x
        save = to_str_tuple(save)
        sources = to_str_tuple(sources)
        targets = to_str_tuple(targets)

        plugins = self._get_plugins(targets, run_id)

        # Get savers/loaders, and meanwhile filter out plugins that do not
        # have to do computation.(their instances will stick around
        # though the .deps attribute of plugins that do)
        loaders = dict()
        savers = collections.defaultdict(list)
        seen = set()
        to_compute = dict()

        def check_cache(d):
            nonlocal plugins, loaders, savers, seen
            if d in seen:
                return
            seen.add(d)
            p = plugins[d]
            key = strax.CacheKey(run_id, d, p.lineage)

            if d not in sources:
                for sb_i, sb in enumerate(self.storage):
                    try:
                        loaders[d] = sb.loader(key)
                        # Found it! No need to make it or save it
                        del plugins[d]
                        return
                    except strax.NotCachedException:
                        continue

                # Not in any cache. We will be computing it.
                to_compute[d] = p
                for d in p.depends_on:
                    check_cache(d)

            # We're making this OR it gets fed in. Should we save it?
            if p.save_when == strax.SaveWhen.NEVER:
                if d in save:
                    raise ValueError("Plugin forbids saving of {d}")
                return
            elif p.save_when == strax.SaveWhen.TARGET:
                if d != targets:
                    return
            elif p.save_when == strax.SaveWhen.EXPLICIT:
                if d not in save:
                    return
            else:
                assert p.save_when == strax.SaveWhen.ALWAYS

            for sb_i, sb in enumerate(self.storage):
                if not sb.provides(d):
                    continue
                s = sb.saver(key, p.metadata(run_id))
                savers[d].append(s)

        for d in targets:
            check_cache(d)
        plugins = to_compute

        # Validate result
        # A data type is either computed, loaded, or fed in
        for a, b in itertools.combinations(
                [plugins.keys(), loaders.keys(), sources], 2):
            if len(a & b):
                raise RuntimeError("Multiple ways of getting "
                                   f"{list(a & b)} specified")

        return strax.ProcessorComponents(
            plugins=plugins,
            loaders=loaders,
            savers=dict(savers),
            sources=sources,
            targets=targets)

    ##
    # Creation of different processors
    ##

    def simple_chain(self, run_id: str, target: str, source: str,
                     save=tuple()):
        components = self.get_components(
            run_id, targets=(target,), save=save, sources=(source,))
        return strax.SimpleChain(components)

    # TODO: fix signatures and docstrings

    def get(self, run_id: str, targets, save=tuple()
            ) -> ty.Iterator[np.ndarray]:
        """Compute target for run_id and iterate over results
        """
        components = self.get_components(run_id, targets=targets, save=save)
        yield from strax.ThreadedMailboxProcessor(
            components, self.executor).iter()

    def make(self, *args, **kwargs):
        for _ in self.get(*args, **kwargs):
            pass

    def get_array(self, *args, **kwargs):
        return np.concatenate(list(self.get(*args, **kwargs)))

    def get_df(self, *args, **kwargs):
        return pd.DataFrame.from_records(self.get_array(*args, **kwargs))

    def in_background(self, run_id, targets, sources=tuple(), save=tuple()):
        """Return a processor that makes targets in a background thread.

        Useful for sending in inputs asynchronously.

        Use as a context manager:
            with strax.in_background(...) as proc:
                proc.send(...)
        """
        components = self.get_components(run_id, targets=targets,
                                         save=save, sources=sources)
        return strax.BackgroundThreadProcessor(components, self.executor)
