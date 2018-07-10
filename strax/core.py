import builtins
import collections
import logging
import fnmatch
import typing as ty
import warnings
import random
import string

import numpy as np
import pandas as pd
import numexpr

import strax
export, __all__ = strax.exporter()


# Placeholder value for omitted values.
# Use instead of None since None might be a proper value/default
OMITTED = '<OMITTED>'


@export
class Context:
    """Context for strax analysis.

    A context holds info on HOW to process data, such as which plugins provide
    what data types, where to store which results, and configuration options
    for the plugins.

    You start all strax processing through a context.
    """
    config: dict

    def __init__(self,
                 storage=None,
                 config=None,
                 register=None,
                 register_all=None):
        """Create a strax context.

        :param storage: Storage front-ends to use. Can be:
          - None (default). Will use DataDirectory('./strax_data').
          - a string: path to use for DataDirectory frontend.
          - list/tuple, or single instance, of storage frontends.
        :param config: Dictionary with configuration options that will be
           applied to plugins
        :param register: plugin class or list of plugin classes to register
        :param register_all: module for which all plugin classes defined in it
           will be registered.
        """
        self.log = logging.getLogger('strax')

        if storage is None:
            storage = ['./strax_data']
        if not isinstance(storage, (list, tuple)):
            storage = [storage]
        self.storage = [strax.DataDirectory(s) if isinstance(s, str) else s
                        for s in storage]

        self._plugin_class_registry = dict()
        self._plugin_instance_cache = dict()

        self.set_config(config, mode='replace')

        if register_all is not None:
            self.register_all(register_all)
        if register is not None:
            self.register(register)

    def new_context(self,
                    storage=tuple(),
                    config=None,
                    register=None,
                    register_all=None,
                    replace=False):
        """Return a new context with new setting adding to those in
        this context.
        :param replace: If True, replaces settings rather than adding them.
        See Context.__init__ for documentation on other parameters.
        """
        # TODO: Clone rather than pass on storage front-ends ??
        if not isinstance(storage, (list, tuple)):
            storage = [storage]
        if config is None:
            config = dict()
        if register is None:
            register = []
        if not isinstance(register, (tuple, list)):
            register = [register]

        if not replace:
            storage = self.storage + list(storage)
            config = {**self.config, **config}
            register = list(self._plugin_class_registry.values()) + register

        return Context(storage=storage,
                       config=config,
                       register=register,
                       register_all=register_all)

    def set_config(self, config=None, mode='update'):
        """Set new configuration options

        :param config: dict of new options
        :param mode: can be either
         - update: Add to or override current options in context
         - setdefault: Add to current options, but do not override
         - replace: Erase config, then set only these options
        """
        if config is None:
            config = dict()
        if mode == 'update':
            self.config.update(config)
        elif mode == 'setdefault':
            for k in config:
                self.config.setdefault(k, config[k])
        elif mode == 'replace':
            self.config = config
        else:
            raise RuntimeError("Expected update, setdefault or new as config"
                               " setting mode")

    def register(self, plugin_class, provides=None):
        """Register plugin_class as provider for data types in provides.
        :param plugin_class: class inheriting from StraxPlugin
        :param provides: list of data types which this plugin provides.

        Plugins always register for the data type specified in the .provide
        class attribute. If such is not available, we will construct one from
        the class name (CamelCase -> snake_case)

        Returns plugin_class (so this can be used as a decorator)
        """
        if isinstance(plugin_class, (tuple, list)) and provides is None:
            # shortcut for multiple registration
            # TODO: document
            for x in plugin_class:
                self.register(x)
            return

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

    def search_field(self, pattern):
        """Find and print which plugin(s) provides a field that matches
        pattern (fnmatch)."""
        cache = dict()
        for d in self._plugin_class_registry:
            if d not in cache:
                cache.update(self._get_plugins((d,), run_id='0'))
            p = cache[d]

            for field_name in p.dtype.names:
                if fnmatch.fnmatch(field_name, pattern):
                    print(f"{field_name} is part of {d} "
                          f"(provided by {p.__class__.__name__})")

    def search_config(self, data_type, pattern='*', run_id='9' * 20):
        """Return configuration options that affect data_type.
        :param data_type: Data type name
        :param pattern: Show only options that match (fnmatch) pattern
        :param run_id: Run id to use for run-dependent config options.
        If omitted, will show defaults active for new runs.
        """
        r = []
        for d, p in self._get_plugins((data_type,), run_id).items():
            for opt in p.takes_config.values():
                if not fnmatch.fnmatch(opt.name, pattern):
                    continue
                try:
                    default = opt.get_default(run_id)
                except strax.InvalidConfiguration:
                    default = OMITTED
                r.append(dict(
                    option=opt.name,
                    default=default,
                    current=self.config.get(opt.name, OMITTED),
                    data_type=d,
                    help=opt.help))
        if len(r):
            return pd.DataFrame(r, columns=r[0].keys())
        return pd.DataFrame([])

    def lineage(self, run_id, data_type):
        """Return lineage dictionary for data_type and run_id, based on the
        options in this context.
        """
        return self._get_plugins((data_type,), run_id)[data_type].lineage

    def register_all(self, module):
        """Register all plugins defined in module"""
        if isinstance(module, (tuple, list)):
            # Secret shortcut for multiple registration
            for x in module:
                self.register_all(x)
            return

        for x in dir(module):
            x = getattr(module, x)
            if type(x) != type(type):
                continue
            if issubclass(x, strax.Plugin):
                self.register(x)

    def data_info(self, data_name: str) -> pd.DataFrame:
        """Return pandas DataFrame describing fields in data_name"""
        p = self._get_plugins((data_name,), run_id='0')[data_name]
        display_headers = ['Field name', 'Data type', 'Comment']
        result = []
        for name, dtype in strax.utils.unpack_dtype(p.dtype):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ''
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    def _set_plugin_config(self, p, run_id, tolerant=True):
        # Explicit type check, since if someone calls this with
        # plugin CLASSES, funny business might ensue
        assert isinstance(p, strax.Plugin)
        config = self.config.copy()
        for opt in p.takes_config.values():
            try:
                opt.validate(config, run_id)
            except strax.InvalidConfiguration:
                if not tolerant:
                    raise
        p.config = {k: v for k, v in config.items()
                    if k in p.takes_config}

    def _get_plugins(self,
                     targets: ty.Tuple[str],
                     run_id: str) -> ty.Dict[str, strax.Plugin]:
        """Return dictionary of plugins necessary to compute targets
        from scratch.
        """
        # Check all config options are taken by some registered plugin class
        # (helps spot typos)
        all_opts = set().union(*[
            pc.takes_config.keys()
            for pc in self._plugin_class_registry.values()])
        for k in self.config:
            if k not in all_opts:
                warnings.warn(f"Option {k} not taken by any registered plugin")

        # Initialize plugins for the entire computation graph
        # (most likely far further down than we need)
        # to get lineages and dependency info.
        def get_plugin(d):
            nonlocal plugins

            if d not in self._plugin_class_registry:
                raise KeyError(f"No plugin class registered that provides {d}")

            plugins[d] = p = self._plugin_class_registry[d]()
            p.run_id = run_id

            # The plugin may not get all the required options here
            # but we don't know if we need the plugin yet
            self._set_plugin_config(p, run_id, tolerant=True)

            p.deps = {d: get_plugin(d) for d in p.depends_on}

            p.lineage = {d: (p.__class__.__name__,
                             p.version(run_id),
                             {q: v for q, v in p.config.items()
                              if p.takes_config[q].track})}
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

            if not isinstance(p, np.dtype):
                dtype = []
                for x in p.dtype:
                    if len(x) == 3:
                        if isinstance(x[0], tuple):
                            # Numpy syntax for array field
                            dtype.append(x)
                        else:
                            # Lazy syntax for normal field
                            field_name, field_type, comment = x
                            dtype.append(((comment, field_name), field_type))
                    elif len(x) == 2:
                        # (field_name, type)
                        dtype.append(x)
                    elif len(x) == 1:
                        # Omitted type: assume float
                        dtype.append((x, np.float))
                    else:
                        raise ValueError(f"Invalid field specification {x}")
                p.dtype = np.dtype(dtype)
            return p

        plugins = collections.defaultdict(get_plugin)
        for t in targets:
            # This works without the LHS too, but your IDE might not get it :-)
            plugins[t] = get_plugin(t)

        return plugins

    def get_components(self, run_id: str,
                       targets=tuple(), save=tuple()
                       ) -> strax.ProcessorComponents:
        """Return components for setting up a processor
        {get_docs}
        """
        save = strax.to_str_tuple(save)
        targets = strax.to_str_tuple(targets)

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
            key = strax.DataKey(run_id, d, p.lineage)

            for sb_i, sf in enumerate(self.storage):
                try:
                    loaders[d] = sf.loader(key)  # TODO: ambiguity options
                    # Found it! No need to make it or save it
                    del plugins[d]
                    return
                except strax.DataNotAvailable:
                    continue

            # Not in any cache. We will be computing it.
            to_compute[d] = p
            for dep_d in p.depends_on:
                check_cache(dep_d)

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

            for sf in self.storage:
                if sf.readonly:
                    continue
                try:
                    savers[d].append(sf.saver(key,
                                              metadata=p.metadata(run_id),
                                              meta_only=p.save_meta_only))
                except strax.DataTypeNotWanted:
                    pass

        for d in targets:
            check_cache(d)
        plugins = to_compute

        intersec = list(plugins.keys() & loaders.keys())
        if len(intersec):
            raise RuntimeError("{intersec} both computed and loaded?!")

        # For the plugins which will run computations,
        # check all required options are available or set defaults.
        # Also run any user-defined setup
        for p in plugins.values():
            self._set_plugin_config(p, run_id, tolerant=False)
            p.setup()
        return strax.ProcessorComponents(
            plugins=plugins,
            loaders=loaders,
            savers=dict(savers),
            targets=targets)

    def get_iter(self, run_id: str, targets, save=tuple(), max_workers=None,
                 selection=None,
                 **kwargs) -> ty.Iterator[np.ndarray]:
        """Compute target for run_id and iterate over results.

        Do NOT interrupt the iterator (i.e. break): it will keep running stuff
        in background threads...
        {get_docs}
        """
        # If any new options given, replace the current context
        # with a temporary one
        if len(kwargs):
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        if isinstance(selection, (list, tuple)):
            selection = ' & '.join(f'({x})' for x in selection)

        # If multiple targets of the same kind, create a MergeOnlyPlugin
        # automatically
        if isinstance(targets, (list, tuple)) and len(targets) > 1:
            plugins = self._get_plugins(targets=targets, run_id=run_id)
            if len(set(plugins[d].data_kind for d in targets)) == 1:
                temp_name = ''.join(random.choices(
                    string.ascii_lowercase, k=10))
                temp_merge = type(temp_name,
                                  (strax.MergeOnlyPlugin,),
                                  dict(depends_on=tuple(targets)))
                self.register(temp_merge)
                targets = temp_name
                # TODO: auto-unregister? Better to have a temp register
                # override option in get_components
            else:
                raise RuntimeError("Cannot automerge different data kinds!")

        components = self.get_components(run_id, targets=targets, save=save)
        for x in strax.ThreadedMailboxProcessor(
                components, max_workers=max_workers).iter():
            if selection is not None:
                mask = numexpr.evaluate(selection, local_dict={
                    fn: x[fn]
                    for fn in x.dtype.names})
                x = x[mask]
            yield x

    def make(self, run_id: str, targets, save=tuple(), max_workers=None,
             **kwargs) -> None:
        """Compute target for run_id. Returns nothing (None).
        {get_docs}
        """
        for _ in self.get_iter(run_id, targets, save, max_workers, **kwargs):
            pass

    def get_array(self, run_id: str, targets, save=tuple(), max_workers=None,
                  **kwargs) -> np.ndarray:
        """Compute target for run_id and return as numpy array
        {get_docs}
        """
        results = list(self.get_iter(run_id, targets, save, max_workers,
                                     **kwargs))
        if len(results):
            return np.concatenate(results)
        raise ValueError("Not a single chunk returned?")

    def get_df(self, run_id: str, targets, save=tuple(), max_workers=None,
               **kwargs) -> pd.DataFrame:
        """Compute target for run_id and return as pandas DataFrame
        {get_docs}
        """
        return pd.DataFrame.from_records(self.get_array(
            run_id, targets, save, max_workers, **kwargs))

    def get_meta(self, run_id, target) -> dict:
        """Return metadata for target for run_id, or raise NotCached
        if data is not yet available.

        :param run_id: run id to get
        :param target: data type to get
        """
        p = self._get_plugins((target,), run_id)[target]
        key = strax.DataKey(run_id, target, p.lineage)
        for sf in self.storage:
            try:
                return sf.get_metadata(key)   # TODO: ambiguity options
            except strax.DataNotAvailable as e:
                print(str(e))
        raise strax.DataNotAvailable(f"Can't load metadata, "
                                     f"data for {key} not available")


get_docs = """
:param run_id: run id to get
:param targets: list/tuple of strings of data type names to get
:param save: extra data types you would like to save
    to cache, if they occur in intermediate computations.
    Many plugins save automatically anyway.
:param max_workers: Number of worker threads/processes to spawn.
    In practice more CPUs may be used due to strax's multithreading.
:param selection: Query string or list of strings with selections to apply.
"""

for attr in dir(Context):
    attr_val = getattr(Context, attr)
    if hasattr(attr_val, '__doc__'):
        doc = attr_val.__doc__
        if doc is not None and '{get_docs}' in doc:
            attr_val.__doc__ = doc.format(get_docs=get_docs)


##
# Config specification. Maybe should be its own file?
##


@export
class InvalidConfiguration(Exception):
    pass


@export
def takes_config(*options):
    """Decorator for plugin classes, to specify which options it takes.
    :param options: Option instances of options this plugin takes.
    """
    def wrapped(plugin_class):
        result = []
        for opt in options:
            if not isinstance(opt, Option):
                raise RuntimeError("Specify config options by Option objects")
            opt.taken_by = plugin_class.__name__
            result.append(opt)

        plugin_class.takes_config = {opt.name: opt for opt in result}
        return plugin_class

    return wrapped


@export
class Option:
    """Configuration option taken by a strax plugin"""
    taken_by: str

    def __init__(self,
                 name: str,
                 type: type = OMITTED,
                 default: ty.Any = OMITTED,
                 default_factory: ty.Callable = OMITTED,
                 default_by_run=OMITTED,
                 track: bool = True,
                 help: str = ''):
        """
        :param name: Option identifier
        :param type: Excepted type of the option's value.
        :param default: Default value the option takes.
        :param default_factory: Function that produces a default value.
        :param default_by_run: Specify that default is run-dependent. Either
         - Callable. Will be called with run_id, must return value for run.
         - List [(start_run_id, value), ..,] for values specified by range of
           runs.
        :param track: If True (default), option value becomes part of plugin
        lineage (just like the plugin version).
        :param help: Human-readable description of the option.
        """
        self.name = name
        self.type = type
        self.default = default
        self.default_by_run = default_by_run
        self.default_factory = default_factory
        self.track = track
        self.help = help

        type = builtins.type
        if sum([self.default is not OMITTED,
                self.default_factory is not OMITTED,
                self.default_by_run is not OMITTED]) > 1:
            raise RuntimeError(f"Tried to specify more than one default "
                               f"for option {self.name}.")

        if type is OMITTED and default is not OMITTED:
            self.type = type(default)

    def get_default(self, run_id):
        """Return default value for the option"""
        if isinstance(run_id, str):
            run_id = int(run_id.replace('_', ''))

        if self.default is not OMITTED:
            return self.default
        if self.default_factory is not OMITTED:
            return self.default_factory()
        if self.default_by_run is not OMITTED:
            if callable(self.default_by_run):
                return self.default_by_run(run_id)
            use_value = OMITTED
            for i, (start_run, value) in enumerate(self.default_by_run):
                if start_run > run_id:
                    break
                use_value = value
            if use_value is OMITTED:
                raise ValueError(
                    f"Run id {run_id} is smaller than the "
                    "lowest run id {start_run} for which the default "
                    "of the option {self.name} is known.")
            return use_value
        raise InvalidConfiguration(f"Missing option {self.name} "
                                   f"required by {self.taken_by}")

    def validate(self, config, run_id, set_defaults=True):
        """Checks if the option is in config and sets defaults if needed.
        """
        if self.name in config:
            value = config[self.name]
            if (self.type is not OMITTED
                    and not isinstance(value, self.type)):
                raise InvalidConfiguration(
                    f"Invalid type for option {self.name}. "
                    f"Excepted a {self.type}, got a {type(value)}")
        elif set_defaults:
            config[self.name] = self.get_default(run_id)
