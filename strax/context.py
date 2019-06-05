from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import logging
import fnmatch
from functools import partial
import random
import re
import string
import typing as ty
import warnings

import numexpr
import numpy as np
import pandas as pd
from tqdm import tqdm

import strax
export, __all__ = strax.exporter()


@strax.takes_config(
    strax.Option(name='storage_converter', default=False,
                 help='If True, save data that is loaded from one frontend '
                      'through all willing other storage frontends.'),
    strax.Option(name='fuzzy_for', default=tuple(),
                 help='Tuple of plugin names for which no checks for version, '
                      'providing plugin, and config will be performed when '
                      'looking for data.'),
    strax.Option(name='fuzzy_for_options', default=tuple(),
                 help='Tuple of config options for which no checks will be '
                      'performed when looking for data.'),
    strax.Option(name='allow_incomplete', default=False,
                 help="Allow loading of incompletely written data, if the "
                      "storage systems support it"),
    strax.Option(name='allow_rechunk', default=True,
                 help="Allow rechunking of data during writing."),
    strax.Option(name='allow_multiprocess', default=False,
                 help="Allow multiprocessing."
                      "If False, will use multithreading only."),
    strax.Option(name='allow_shm', default=False,
                 help="Allow use of /dev/shm for interprocess communication."),
    strax.Option(name='forbid_creation_of', default=tuple(),
                 help="If any of the following datatypes is requested to be "
                      "created, throw an error instead. Useful to limit "
                      "descending too far into the dependency graph."),
    strax.Option(name='store_run_fields', default=tuple(),
                 help="Tuple of run document fields to store "
                      "during scan_run."),
    strax.Option(name='check_available', default=tuple(),
                 help="Tuple of data types to scan availability for "
                      "during scan_run."))
@export
class Context:
    """Context for strax analysis.

    A context holds info on HOW to process data, such as which plugins provide
    what data types, where to store which results, and configuration options
    for the plugins.

    You start all strax processing through a context.
    """
    config: dict
    context_config: dict

    runs: ty.Union[pd.DataFrame, type(None)] = None

    def __init__(self,
                 storage=None,
                 config=None,
                 register=None,
                 register_all=None,
                 **kwargs):
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
        Any additional kwargs are considered Context-specific options; see
        Context.takes_config.
        """
        self.log = logging.getLogger('strax')

        if storage is None:
            storage = ['./strax_data']
        if not isinstance(storage, (list, tuple)):
            storage = [storage]
        self.storage = [strax.DataDirectory(s) if isinstance(s, str) else s
                        for s in storage]

        self._plugin_class_registry = dict()

        self.set_config(config, mode='replace')
        self.set_context_config(kwargs, mode='replace')

        if register_all is not None:
            self.register_all(register_all)
        if register is not None:
            self.register(register)

    def new_context(self,
                    storage=tuple(),
                    config=None,
                    register=None,
                    register_all=None,
                    replace=False,
                    **kwargs):
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
            config = strax.combine_configs(self.config,
                                           config,
                                           mode='update')
            kwargs = strax.combine_configs(self.context_config,
                                           kwargs,
                                           mode='update')
            register = list(self._plugin_class_registry.values()) + register

        return Context(storage=storage,
                       config=config,
                       register=register,
                       register_all=register_all,
                       **kwargs)

    def set_config(self, config=None, mode='update'):
        """Set new configuration options

        :param config: dict of new options
        :param mode: can be either
         - update: Add to or override current options in context
         - setdefault: Add to current options, but do not override
         - replace: Erase config, then set only these options
        """
        if not hasattr(self, 'config'):
            self.config = dict()
        self.config = strax.combine_configs(
            old_config=self.config,
            new_config=config,
            mode=mode)

    def set_context_config(self, context_config=None, mode='update'):
        """Set new context configuration options

        :param context_config: dict of new context configuration options
        :param mode: can be either
         - update: Add to or override current options in context
         - setdefault: Add to current options, but do not override
         - replace: Erase config, then set only these options
        """
        if not hasattr(self, 'context_config'):
            self.context_config = dict()

        new_config = strax.combine_configs(
            old_config=self.context_config,
            new_config=context_config,
            mode=mode)

        for opt in self.takes_config.values():
            opt.validate(new_config, set_defaults=True)

        for k in new_config:
            if k not in self.takes_config:
                warnings.warn(f"Unknown config option {k}; will do nothing.")

        self.context_config = new_config

        for k in self.context_config:
            if k not in self.takes_config:
                warnings.warn(f"Invalid context option {k}; will do nothing.")

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

    def show_config(self, data_type=None, pattern='*', run_id='9' * 20):
        """Return configuration options that affect data_type.
        :param data_type: Data type name
        :param pattern: Show only options that match (fnmatch) pattern
        :param run_id: Run id to use for run-dependent config options.
        If omitted, will show defaults active for new runs.
        """
        r = []
        if data_type is None:
            # search for context options
            it = [['Context', self]]
        else:
            it = self._get_plugins((data_type,), run_id).items()
        for d, p in it:
            for opt in p.takes_config.values():
                if not fnmatch.fnmatch(opt.name, pattern):
                    continue
                try:
                    default = opt.get_default(run_id)
                except strax.InvalidConfiguration:
                    default = strax.OMITTED
                c = self.context_config if data_type is None else self.config
                r.append(dict(
                    option=opt.name,
                    default=default,
                    current=c.get(opt.name, strax.OMITTED),
                    applies_to=d,
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

    @property
    def _find_options(self):
        return dict(fuzzy_for=self.context_config['fuzzy_for'],
                    fuzzy_for_options=self.context_config['fuzzy_for_options'],
                    allow_incomplete=self.context_config['allow_incomplete'])

    def get_components(self, run_id: str,
                       targets=tuple(), save=tuple(),
                       time_range=None,
                      ) -> strax.ProcessorComponents:
        """Return components for setting up a processor
        {get_docs}
        """
        save = strax.to_str_tuple(save)
        targets = strax.to_str_tuple(targets)

        plugins = self._get_plugins(targets, run_id)

        n_range = None
        if time_range is not None:
            # Ensure we have one data kind
            if len(set([plugins[t].data_kind for t in targets])) > 1:
                raise NotImplementedError(
                    "Time range selection not implemented "
                    "for multiple data kinds.")

            # Which plugin provides time information? We need it to map to
            # row indices.
            for p in targets:
                if 'time' in plugins[p].dtype.names:
                    break
            else:
                raise RuntimeError(f"No time info in targets, should have been"
                                   f" caught earlier??")

            # Find a range of row numbers that contains the time range
            # It's a bit too large: to
            # Get the n <-> time mapping in needed chunks
            if not self.is_stored(run_id, p):
                raise strax.DataNotAvailable(f"Time range selection needs time"
                                             f" info from {p}, but this data"
                                             f" is not yet available")
            meta = self.get_meta(run_id, p)
            times = np.array([c['first_time'] for c in meta['chunks']])
            # Reconstruct row numbers from row counts, which are in metadata
            # n_end is last row + 1 in a chunk. n_start is the first.
            n_end = np.array([c['n'] for c in meta['chunks']]).cumsum()
            n_start = n_end - n_end[0]
            _inds = np.searchsorted(times, time_range) - 1
            # Clip to prevent out-of-range times causing
            # negative or nonexistent indices
            _inds = np.clip(_inds, 0, len(n_end) - 1)
            n_range = n_start[_inds[0]], n_end[_inds[1]]

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
                    # Bit clunky... but allows specifying executor later
                    sf.find(key, **self._find_options)
                    loaders[d] = partial(sf.loader,
                        key,
                        n_range=n_range,
                        **self._find_options)
                    # Found it! No need to make it
                    del plugins[d]
                    break
                except strax.DataNotAvailable:
                    continue
            else:
                if time_range is not None:
                    # While the data type providing the time information is
                    # available (else we'd have failed earlier), one of the
                    # other requested data types is not.
                    raise strax.DataNotAvailable(
                        f"Time range selection assumes data is already "
                        f"available, but {d} for {run_id} is not.")
                if d in self.context_config['forbid_creation_of']:
                    raise strax.DataNotAvailable(
                        f"{d} for {run_id} not found in any storage, and "
                        "your context specifies it cannot be created.")
                # Not in any cache. We will be computing it.
                to_compute[d] = p
                for dep_d in p.depends_on:
                    check_cache(dep_d)

            # Should we save this data?
            if time_range is not None:
                # No, since we're not even getting the whole data.
                # Without this check, saving could be attempted if the
                # storage converter mode is enabled.
                self.log.warning(f"Not saving {d} while "
                                 f"selecting a time range in the run")
                return
            if any([len(v) > 0
                    for k, v in self._find_options.items()
                    if 'fuzzy' in k]):
                # In fuzzy matching mode, we cannot (yet) derive the lineage
                # of any data we are creating. To avoid create false
                # data entries, we currently do not save at all.
                self.log.warning(f"Not saving {d} while fuzzy matching is "
                                 f"turned on.")
                return
            if self.context_config['allow_incomplete']:
                self.log.warning(f"Not saving {d} while loading incomplete "
                                 f"data is allowed.")
                return

            elif p.save_when == strax.SaveWhen.NEVER:
                if d in save:
                    raise ValueError("Plugin forbids saving of {d}")
                return
            elif p.save_when == strax.SaveWhen.TARGET:
                if d not in targets:
                    return
            elif p.save_when == strax.SaveWhen.EXPLICIT:
                if d not in save:
                    return
            else:
                assert p.save_when == strax.SaveWhen.ALWAYS

            for sf in self.storage:
                if sf.readonly:
                    continue
                if d not in to_compute:
                    if not self.context_config['storage_converter']:
                        continue
                    try:
                        sf.find(key,
                                **self._find_options)
                        # Already have this data in this backend
                        continue
                    except strax.DataNotAvailable:
                        # Don't have it, so let's convert it!
                        pass
                try:
                    savers[d].append(sf.saver(key,
                                              metadata=p.metadata(run_id)))
                except strax.DataNotAvailable:
                    # This frontend cannot save. Too bad.
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

    def get_iter(self, run_id: str,
                 targets, save=tuple(), max_workers=None,
                 time_range=None,
                 seconds_range=None,
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

        # Convert relative to absolute time range
        if seconds_range is not None:
            try:
                # Use run metadata, if it is available, to get
                # the run start time (floored to seconds)
                t0 = self.run_metadata(run_id, 'start')['start']
                t0 = int(t0.timestamp()) * int(1e9)
            except Exception:
                # Get an approx start from the data itself,
                # then floor it to seconds for consistency
                if isinstance(targets, (list, tuple)):
                    t = targets[0]
                else:
                    t = targets
                t0 = self.get_meta(run_id, t)['chunks'][0]['first_time']
                t0 = int(t0 / int(1e9)) * int(1e9)
            time_range = (t0 + int(1e9) * seconds_range[0],
                          t0 + int(1e9) * seconds_range[1])

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
                # Or just always create new context, not only if new options
                # are given
            else:
                raise RuntimeError("Cannot automerge different data kinds!")

        components = self.get_components(run_id, targets=targets, save=save,
                                         time_range=time_range)
        for x in strax.ThreadedMailboxProcessor(
                components,
                max_workers=max_workers,
                allow_shm=self.context_config['allow_shm'],
                allow_multiprocess=self.context_config['allow_multiprocess'],
                allow_rechunk=self.context_config['allow_rechunk']).iter():
            if selection is not None:
                mask = numexpr.evaluate(selection, local_dict={
                    fn: x[fn]
                    for fn in x.dtype.names})
                x = x[mask]
            if time_range:
                if 'time' not in x.dtype.names:
                    raise NotImplementedError(
                        "Time range selection requires time information, "
                        "but none of the required plugins provides it.")
                x = x[(time_range[0] <= x['time']) &
                      (x['time'] < time_range[1])]
            yield x

    def make(self, run_id: ty.Union[str, tuple, list],
             targets, save=tuple(), max_workers=None,
             **kwargs) -> None:
        """Compute target for run_id. Returns nothing (None).
        {get_docs}
        """
        # Multi-run support
        run_ids = strax.to_str_tuple(run_id)
        if len(run_ids) > 1:
            return multi_run(self, run_ids, targets=targets,
                             save=save, max_workers=max_workers, **kwargs)

        for _ in self.get_iter(run_ids[0], targets,
                               save=save, max_workers=max_workers, **kwargs):
            pass

    def get_array(self, run_id: ty.Union[str, tuple, list],
                  targets, save=tuple(), max_workers=None,
                  **kwargs) -> np.ndarray:
        """Compute target for run_id and return as numpy array
        {get_docs}
        """
        run_ids = strax.to_str_tuple(run_id)
        if len(run_ids) > 1:
            results = multi_run(self.get_array, run_ids, targets=targets,
                                save=save, max_workers=max_workers, **kwargs)
        else:
            results = list(self.get_iter(run_ids[0], targets,
                                         save=save, max_workers=max_workers,
                                         **kwargs))
        if len(results):
            return np.concatenate(results)
        raise ValueError("No results returned?")

    def get_df(self, run_id: ty.Union[str, tuple, list],
               targets, save=tuple(), max_workers=None,
               **kwargs) -> pd.DataFrame:
        """Compute target for run_id and return as pandas DataFrame
        {get_docs}
        """
        df = self.get_array(
            run_id, targets,
            save=save, max_workers=max_workers, **kwargs)
        try:
            return pd.DataFrame.from_records(df)
        except Exception as e:
            if 'Data must be 1-dimensional' in str(e):
                raise ValueError(
                    f"Cannot load '{targets}' as a dataframe because it has "
                    f"array fields. Please use get_array.")
            raise

    def _key_for(self, run_id, target):
        p = self._get_plugins((target,), run_id)[target]
        return strax.DataKey(run_id, target, p.lineage)

    def get_meta(self, run_id, target) -> dict:
        """Return metadata for target for run_id, or raise DataNotAvailable
        if data is not yet available.

        :param run_id: run id to get
        :param target: data type to get
        """
        key = self._key_for(run_id, target)
        for sf in self.storage:
            try:
                return sf.get_metadata(key, **self._find_options)
            except strax.DataNotAvailable as e:
                self.log.debug(f"Frontend {sf} does not have {key}")
        raise strax.DataNotAvailable(f"Can't load metadata, "
                                     f"data for {key} not available")

    get_metadata = get_meta

    def run_metadata(self, run_id, projection=None) -> dict:
        """Return run-level metadata for run_id, or raise DataNotAvailable
        if this is not available

        :param run_id: run id to get
        :param projection: Selection of fields to get, following MongoDB
        syntax. May not be supported by frontend.
        """
        for sf in self.storage:
            if not sf.provide_run_metadata:
                continue
            try:
                return sf.run_metadata(run_id, projection=projection)
            except (strax.DataNotAvailable, NotImplementedError):
                self.log.debug(f"Frontend {sf} does not have "
                               f"run metadata for {run_id}")
        raise strax.DataNotAvailable(f"No run-level metadata available "
                                     f"for {run_id}")

    def is_stored(self, run_id, target, **kwargs):
        """Return whether data type target has been saved for run_id
        through any of the registered storage frontends.

        Note that even if False is returned, the data type may still be made
        with a trivial computation.
        """
        # If any new options given, replace the current context
        # with a temporary one
        # TODO duplicated code with with get_iter
        if len(kwargs):
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        key = self._key_for(run_id, target)
        for sf in self.storage:
            try:
                sf.find(key, **self._find_options)
                return True
            except strax.DataNotAvailable:
                continue
        return False

    def list_available(self, target, **kwargs):
        """Return sorted list of run_id's for which target is available
        """
        # TODO duplicated code with with get_iter
        if len(kwargs):
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        if self.runs is None:
            self.scan_runs()

        keys = set([
            self._key_for(run_id, target)
            for run_id in self.runs['name'].values])

        found = set()
        for sf in self.storage:
            remaining = keys - found
            is_found = sf.find_several(remaining, **self._find_options)
            found |= set([k for i, k in enumerate(remaining)
                          if is_found[i]])
        return list(sorted([x.run_id for x in found]))

    def scan_runs(self,
                  check_available=tuple(),
                  store_fields=tuple()):
        """Update and return self.runs with runs currently available
        in all storage frontends.
        :param check_available: Check whether these data types are available
        Availability of xxx is stored as a boolean in the xxx_available
        column.
        :param store_fields: Additional fields from run doc to include
        as rows in the dataframe.

        The context options scan_availability and store_run_fields list
        data types and run fields, respectively, that will always be scanned.
        """
        store_fields = tuple(set(
            list(strax.to_str_tuple(store_fields))
            + ['name', 'number', 'tags', 'mode']
            + list(self.context_config['store_run_fields'])))
        check_available = tuple(set(
            list(strax.to_str_tuple(check_available))
            + list(self.context_config['check_available'])))

        docs = None
        for sf in self.storage:
            _temp_docs = []
            for doc in sf._scan_runs(store_fields=store_fields):
                # If there is no number, make one from the name
                if 'number' not in doc:
                    if 'name' not in doc:
                        raise ValueError(f"Invalid run doc {doc}, contains "
                                         f"neither name nor number.")
                    doc['number'] = int(doc['name'])

                # If there is no name, make one from the number
                doc.setdefault('name', str(doc['number']))

                doc.setdefault('mode', '')

                # Flatten the tags field, if it exists
                doc['tags'] = ','.join([t['name']
                                        for t in doc.get('tags', [])])

                # Flatten the rest of the doc (mainly in case the mode field
                # is something deeply nested)
                doc = strax.flatten_dict(doc, separator='.')

                _temp_docs.append(doc)

            if len(_temp_docs):
                new_docs = pd.DataFrame(_temp_docs)
            else:
                new_docs = pd.DataFrame([], columns=store_fields)

            if docs is None:
                docs = new_docs
            else:
                # Keep only new runs (not found by earlier frontends)
                docs = pd.concat([
                    docs,
                    new_docs[
                        ~np.in1d(new_docs['name'], docs['name'])]],
                    sort=False)

        self.runs = docs

        for d in tqdm(check_available,
                      desc='Checking data availability'):
            self.runs[d + '_available'] = np.in1d(
                self.runs.name.values,
                self.list_available(d))

        return self.runs

    def select_runs(self, run_mode=None,
                    include_tags=None, exclude_tags=None,
                    available=tuple(),
                    pattern_type='fnmatch', ignore_underscore=True):
        """Return pandas.DataFrame with basic info from runs
        that match selection criteria.
        :param run_mode: Pattern to match run modes (reader.ini.name)
        :param available: str or tuple of strs of data types for which data
        must be available according to the runs DB.

        :param include_tags: String or list of strings of patterns
            for required tags
        :param exclude_tags: String / list of strings of patterns
            for forbidden tags.
            Exclusion criteria  have higher priority than inclusion criteria.
        :param pattern_type: Type of pattern matching to use.
            Defaults to 'fnmatch', which means you can use
            unix shell-style wildcards (`?`, `*`).
            The alternative is 're', which means you can use
            full python regular expressions.
        :param ignore_underscore: Ignore the underscore at the start of tags
            (indicating some degree of officialness or automation).

        Examples:
         - `run_selection(include_tags='blinded')`
            select all datasets with a blinded or _blinded tag.
         - `run_selection(include_tags='*blinded')`
            ... with blinded or _blinded, unblinded, blablinded, etc.
         - `run_selection(include_tags=['blinded', 'unblinded'])`
            ... with blinded OR unblinded, but not blablinded.
         - `run_selection(include_tags='blinded',
                          exclude_tags=['bad', 'messy'])`
           select blinded dsatasets that aren't bad or messy
        """
        if self.runs is None:
            self.scan_runs()
        dsets = self.runs.copy()

        if pattern_type not in ('re', 'fnmatch'):
            raise ValueError("Pattern type must be 're' or 'fnmatch'")

        if run_mode is not None:
            modes = dsets['mode'].values
            mask = np.zeros(len(modes), dtype=np.bool_)
            if pattern_type == 'fnmatch':
                for i, x in enumerate(modes):
                    mask[i] = fnmatch.fnmatch(x, run_mode)
            elif pattern_type == 're':
                for i, x in enumerate(modes):
                    mask[i] = bool(re.match(run_mode, x))
            dsets = dsets[mask]

        if include_tags is not None:
            dsets = dsets[_tags_match(dsets,
                                      include_tags,
                                      pattern_type,
                                      ignore_underscore)]

        if exclude_tags is not None:
            dsets = dsets[True ^ _tags_match(dsets,
                                             exclude_tags,
                                             pattern_type,
                                             ignore_underscore)]

        have_available = strax.to_str_tuple(available)
        for d in have_available:
            if not d + '_available' in dsets.columns:
                # Get extra availability info from the run db
                self.runs[d + '_available'] = np.in1d(
                    self.runs.name.values,
                    self.list_available(d))
            dsets = dsets[dsets[d + '_available']]

        return dsets

    def define_run(self,
                   name: str,
                   data: ty.Union[np.ndarray, pd.DataFrame, dict],
                   from_run: ty.Union[str, None] = None):

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            # Array of events / regions of interest
            start, end = data['time'], strax.endtime(data)
            if from_run is not None:
                return self.define_run(
                    name,
                    {from_run: np.transpose([start, end])})
            else:
                df = pd.DataFrame(dict(starts=start, ends=end,
                                       run_id=data['run_id']))
                self.define_run(
                    name,
                    {run_id: rs[['start', 'stop']].values.transpose()
                     for run_id, rs in df.groupby('fromrun')})

        if isinstance(data, (list, tuple)):
            # list of runids
            data = strax.to_str_tuple(data)
            self.define_run(
                name,
                {run_id: 'all' for run_id in data})

        if not isinstance(data, dict):
            raise ValueError("Can't define run from {type(data)}")

        # Dict mapping run_id: array of time ranges or all
        for sf in self.storage:
            if not sf.readonly and sf.can_define_runs:
                sf.define_run(name, data)
                break
        else:
            raise RuntimeError("No storage frontend registered that allows"
                               " run definition")



get_docs = """
:param run_id: run id to get
:param targets: list/tuple of strings of data type names to get
:param save: extra data types you would like to save
    to cache, if they occur in intermediate computations.
    Many plugins save automatically anyway.
:param max_workers: Number of worker threads/processes to spawn.
    In practice more CPUs may be used due to strax's multithreading.
:param selection: Query string or list of strings with selections to apply.
:param time_range: (start, stop) range of ns since the unix epoch to load
:param seconds_range: (start, stop) range of seconds since the start of the
run to load.
"""

for attr in dir(Context):
    attr_val = getattr(Context, attr)
    if hasattr(attr_val, '__doc__'):
        doc = attr_val.__doc__
        if doc is not None and '{get_docs}' in doc:
            attr_val.__doc__ = doc.format(get_docs=get_docs)


def _tags_match(dsets, patterns, pattern_type, ignore_underscore):
    result = np.zeros(len(dsets), dtype=np.bool)

    if isinstance(patterns, str):
        patterns = [patterns]

    for i, tags in enumerate(dsets.tags):
        result[i] = any([any([_tag_match(tag, pattern,
                                         pattern_type,
                                         ignore_underscore)
                              for tag in tags.split(',')
                              for pattern in patterns])])

    return result


def _tag_match(tag, pattern, pattern_type, ignore_underscore):
    if ignore_underscore and tag.startswith('_'):
        tag = tag[1:]
    if pattern_type == 'fnmatch':
        return fnmatch.fnmatch(tag, pattern)
    elif pattern_type == 're':
        return bool(re.match(pattern, tag))
    raise NotImplementedError


@export
def multi_run(f, run_ids, *args, max_workers=None, **kwargs):
    """Execute f(run_id, **kwargs) over multiple runs,
    then return list of results.

    :param run_ids: list/tuple of runids
    :param max_workers: number of worker threads/processes to spawn

    Other (kw)args will be passed to f
    """
    # Try to int all run_ids

    # Get a numpy array of run ids.
    try:
        run_id_numpy = np.array([int(x) for x in run_ids],
                                dtype=np.int32)
    except ValueError:
        # If there are string id's among them,
        # numpy will autocast all the run ids to Unicode fixed-width
        run_id_numpy = np.array(run_ids)

    # Probably we'll want to use dask for this in the future,
    # to enable cut history tracking and multiprocessing.
    # For some reason the ProcessPoolExecutor doesn't work??
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = [exc.submit(f, r, *args, **kwargs)
                   for r in run_ids]
        for _ in tqdm(as_completed(futures),
                      desc="Loading %d runs" % len(run_ids)):
            pass

        result = []
        for i, f in enumerate(futures):
            r = f.result()
            ids = np.array([run_id_numpy[i]] * len(r),
                           dtype=[('run_id', run_id_numpy.dtype)])
            r = strax.merge_arrs([ids, r])
            result.append(r)
        return result
