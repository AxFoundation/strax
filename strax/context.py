import collections
import datetime
import logging
import fnmatch
from functools import partial
import random
import string
import typing as ty
import warnings

import contextlib
import sys
if any('jupyter' in arg for arg in sys.argv):
    # In some cases we are not using any notebooks,
    # Taken from 44952863 on stack overflow thanks!
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import numexpr
import numpy as np
import pandas as pd

import strax
export, __all__ = strax.exporter()
__all__ += ['RUN_DEFAULTS_KEY']

RUN_DEFAULTS_KEY = 'strax_defaults'


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
    strax.Option(name='allow_lazy', default=True,
                 help='Allow "lazy" processing. Saves memory, but incompatible '
                      'with multiprocessing and perhaps slightly slower.'),
    strax.Option(name='forbid_creation_of', default=tuple(),
                 help="If any of the following datatypes is requested to be "
                      "created, throw an error instead. Useful to limit "
                      "descending too far into the dependency graph."),
    strax.Option(name='store_run_fields', default=tuple(),
                 help="Tuple of run document fields to store "
                      "during scan_run."),
    strax.Option(name='check_available', default=tuple(),
                 help="Tuple of data types to scan availability for "
                      "during scan_run."),
    strax.Option(name='max_messages', default=4,
                 help="Maximum number of mailbox messages, i.e. size of buffer "
                      "between plugins. Too high = RAM blows up. "
                      "Too low = likely deadlocks."),
    strax.Option(name='timeout', default=24 * 3600,
                 help="Terminate processing if any one mailbox receives "
                      "no result for more than this many seconds"),
    strax.Option(name='use_per_run_defaults', default=False,
                 help='Scan the run db for per-run defaults. '
                      'This is an experimental strax feature that will '
                      'possibly be removed, see issue #246'),
    strax.Option(name='free_options', default=tuple(),
                 help='Do not warn if any of these options are passed, '
                      'even when no registered plugin takes them.')
)
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
    _run_defaults_cache: dict = None

    storage: ty.List[strax.StorageFrontend]

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
        self._run_defaults_cache = dict()

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

        new_c = Context(storage=storage, config=config, **kwargs)
        if not replace:
            new_c._plugin_class_registry = self._plugin_class_registry.copy()
        new_c.register_all(register_all)
        new_c.register(register)
        return new_c

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
            opt.validate(new_config)

        for k in new_config:
            if k not in self.takes_config:
                warnings.warn(f"Unknown config option {k}; will do nothing.")

        self.context_config = new_config

        for k in self.context_config:
            if k not in self.takes_config:
                warnings.warn(f"Invalid context option {k}; will do nothing.")

    def register(self, plugin_class):
        """Register plugin_class as provider for data types in provides.
        :param plugin_class: class inheriting from strax.Plugin.
        You can also pass a sequence of plugins to register, but then
        you must omit the provides argument.

        If a plugin class omits the .provides attribute, we will construct
        one from its class name (CamelCase -> snake_case)

        Returns plugin_class (so this can be used as a decorator)
        """
        if isinstance(plugin_class, (tuple, list)):
            # Shortcut for multiple registration
            for x in plugin_class:
                self.register(x)
            return

        if not hasattr(plugin_class, 'provides'):
            # No output name specified: construct one from the class name
            snake_name = strax.camel_to_snake(plugin_class.__name__)
            plugin_class.provides = (snake_name,)

        # Ensure plugin_class.provides is a tuple
        if isinstance(plugin_class.provides, str):
            plugin_class.provides = tuple([plugin_class.provides])

        for p in plugin_class.provides:
            self._plugin_class_registry[p] = plugin_class

        already_seen = []
        for plugin in self._plugin_class_registry.values():

            if plugin in already_seen:
                continue
            already_seen.append(plugin)

            for option, items in plugin.takes_config.items():
                try:
                    # Looping over the options of the new plugin and check if
                    # they can be found in the already registered plugins:
                    for new_option, new_items in plugin_class.takes_config.items():
                        if not new_option == option:
                            continue
                        default = items.get_default('0')  # Have to pass will be changed.
                        new_default = new_items.get_default('0')
                        if default == new_default:
                            continue
                        else:
                            mes = (f'Two plugins have a different default value'
                                   f' for the same option. The option'
                                   f' "{new_option}" in "{plugin.__name__}" takes'
                                   f' as a default "{default}"  while in'
                                   f' "{plugin_class.__name__}" the default value'
                                   f' is set to "{new_default}". Please change'
                                   ' one of the defaults.'
                                   )
                            raise ValueError(mes)

                except strax.InvalidConfiguration:
                    # These are option which are inherited from context options.
                    pass

        return plugin_class

    def search_field(self, pattern):
        """Find and print which plugin(s) provides a field that matches
        pattern (fnmatch)."""
        cache = dict()
        for d in self._plugin_class_registry:
            if d not in cache:
                cache.update(self._get_plugins((d,), run_id='0'))
            p = cache[d]

            for field_name in p.dtype_for(d).fields:
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
        seen = []
        for d, p in it:
            # Track plugins we already saw, so options from
            # multi-output plugins don't come up several times
            if p in seen:
                continue
            seen.append(p)

            for opt in p.takes_config.values():
                if not fnmatch.fnmatch(opt.name, pattern):
                    continue
                try:
                    default = opt.get_default(run_id, self.run_defaults(run_id))
                except strax.InvalidConfiguration:
                    default = strax.OMITTED
                c = self.context_config if data_type is None else self.config
                r.append(dict(
                    option=opt.name,
                    default=default,
                    current=c.get(opt.name, strax.OMITTED),
                    applies_to=(p.provides if d != 'Context' else d),
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
        """Register all plugins defined in module.

        Can pass a list/tuple of modules to register all in each.
        """
        if isinstance(module, (tuple, list)):
            # Shortcut for multiple registration
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
        for name, dtype in strax.utils.unpack_dtype(p.dtype_for(data_name)):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ''
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    def get_single_plugin(self, run_id, data_name):
        """Return a single fully initialized plugin that produces
        data_name for run_id. For use in custom processing."""
        plugin = self._get_plugins((data_name,), run_id)[data_name]
        self._set_plugin_config(plugin, run_id, tolerant=False)
        plugin.setup()
        return plugin

    def _set_plugin_config(self, p, run_id, tolerant=True):
        # Explicit type check, since if someone calls this with
        # plugin CLASSES, funny business might ensue
        assert isinstance(p, strax.Plugin)
        config = self.config.copy()
        for opt in p.takes_config.values():
            try:
                opt.validate(config,
                             run_id=run_id,
                             run_defaults=self.run_defaults(run_id))
            except strax.InvalidConfiguration:
                if not tolerant:
                    raise
        p.config = {k: v for k, v in config.items()
                    if k in p.takes_config}

    def _get_plugins(self,
                     targets: ty.Tuple[str],
                     run_id: str) -> ty.Dict[str, strax.Plugin]:
        """Return dictionary of plugin instances necessary to compute targets
        from scratch.
        For a plugin that produces multiple outputs, we make only a single
        instance, which is referenced under multiple keys in the output dict.
        """
        # Check all config options are taken by some registered plugin class
        # (helps spot typos)
        all_opts = set().union(*[
            pc.takes_config.keys()
            for pc in self._plugin_class_registry.values()])
        for k in self.config:
            if not (k in all_opts or k in self.context_config['free_options']):
                warnings.warn(f"Option {k} not taken by any registered plugin")

        # Initialize plugins for the entire computation graph
        # (most likely far further down than we need)
        # to get lineages and dependency info.
        def get_plugin(data_kind):
            nonlocal plugins

            if data_kind not in self._plugin_class_registry:
                raise KeyError(f"No plugin class registered that provides {data_kind}")

            p = self._plugin_class_registry[data_kind]()

            d_provides = None  # just to make codefactor happy
            for d_provides in p.provides:
                plugins[d_provides] = p

            p.run_id = run_id

            # The plugin may not get all the required options here
            # but we don't know if we need the plugin yet
            self._set_plugin_config(p, run_id, tolerant=True)

            p.deps = {d_depends: get_plugin(d_depends) for d_depends in p.depends_on}

            last_provide = d_provides
            p.lineage = {last_provide: (p.__class__.__name__,
                             p.version(run_id),
                             {q: v for q, v in p.config.items()
                              if p.takes_config[q].track})}
            for d_depends in p.depends_on:
                p.lineage.update(p.deps[d_depends].lineage)

            if not hasattr(p, 'data_kind') and not p.multi_output:
                if len(p.depends_on):
                    # Assume data kind is the same as the first dependency
                    first_dep = p.depends_on[0]
                    p.data_kind = p.deps[first_dep].data_kind_for(first_dep)
                else:
                    # No dependencies: assume provided data kind and
                    # data type are synonymous
                    p.data_kind = p.provides[0]

            p.fix_dtype()

            return p

        plugins = {}
        for t in targets: 
            p = get_plugin(t)
            plugins[t] = p

        return plugins

    @property
    def _find_options(self):
        return dict(fuzzy_for=self.context_config['fuzzy_for'],
                    fuzzy_for_options=self.context_config['fuzzy_for_options'],
                    allow_incomplete=self.context_config['allow_incomplete'])

    def _get_partial_loader_for(self, key, time_range=None, chunk_number=None):
        """
        Get partial loaders to allow loading data later
        :param key: strax.DataKey
        :param time_range: 2-length arraylike of (start, exclusive end) of row
        numbers to get. Default is None, which means get the entire run.
        :param chunk_number: number of the chunk for data specified by
        strax.DataKey. This chunck is loaded exclusively.
        :return: partial object
        """
        for sb_i, sf in enumerate(self.storage):
            try:
                # Partial is clunky... but allows specifying executor later
                # Since it doesn't run until later, we must do a find now
                # that we can still handle DataNotAvailable
                sf.find(key, **self._find_options)
                return partial(sf.loader,
                               key,
                               time_range=time_range,
                               chunk_number=chunk_number,
                               **self._find_options)
            except strax.DataNotAvailable:
                continue
        return False

    def get_components(self, run_id: str,
                       targets=tuple(), save=tuple(),
                       time_range=None, chunk_number=None
                       ) -> strax.ProcessorComponents:
        """Return components for setting up a processor
        {get_docs}
        """

        save = strax.to_str_tuple(save)
        targets = strax.to_str_tuple(targets)

        # Although targets is a tuple, we only support one target at the moment
        # we could just make it a string!
        assert len(targets) == 1, f"Found {len(targets)} instead of 1 target"
        if len(targets[0]) == 1:
            raise ValueError(
                f"Plugin names must be more than one letter, not {targets[0]}")

        plugins = self._get_plugins(targets, run_id)
        target = targets[0]  # See above, already restricted to one target

        # Get savers/loaders, and meanwhile filter out plugins that do not
        # have to do computation. (their instances will stick around
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

            # Can we load this data?
            loading_this_data = False
            key = strax.DataKey(run_id, d, p.lineage)

            ldr = self._get_partial_loader_for(
                key,
                chunk_number=chunk_number,
                time_range=time_range)

            if not ldr and run_id.startswith('_'):
                if time_range is not None:
                    raise NotImplementedError("time range loading not yet "
                                              "supported for superruns")

                sub_run_spec = self.run_metadata(
                    run_id, 'sub_run_spec')['sub_run_spec']
                self.make(list(sub_run_spec.keys()), d)

                ldrs = []
                for subrun in sub_run_spec:
                    sub_key = strax.DataKey(
                        subrun,
                        d,
                        self._get_plugins((d,), subrun)[d].lineage)
                    if sub_run_spec[subrun] == 'all':
                        _subrun_time_range = None
                    else:
                        _subrun_time_range = sub_run_spec[subrun]
                    ldr = self._get_partial_loader_for(
                        sub_key,
                        time_range=_subrun_time_range,
                        chunk_number=chunk_number)
                    if not ldr:
                        raise RuntimeError(
                            f"Could not load {d} for subrun {subrun} "
                            f"even though we made it??")
                    ldrs.append(ldr)

                def concat_loader(*args, **kwargs):
                    for x in ldrs:
                        yield from x(*args, **kwargs)
                ldr = lambda *args, **kwargs : concat_loader(*args, **kwargs)

            if ldr:
                # Found it! No need to make it or look in other frontends
                loading_this_data = True
                loaders[d] = ldr
                del plugins[d]
            else:
                # Data not found anywhere. We will be computing it.
                self._check_forbidden()
                if (time_range is not None
                        and plugins[d].save_when != strax.SaveWhen.NEVER):
                    # While the data type providing the time information is
                    # available (else we'd have failed earlier), one of the
                    # other requested data types is not.
                    raise strax.DataNotAvailable(
                        f"Time range selection assumes data is already "
                        f"available, but {d} for {run_id} is not.")
                if '*' in self.context_config['forbid_creation_of']:
                    raise strax.DataNotAvailable(
                        f"{d} for {run_id} not found in any storage, and "
                        "your context specifies no new data can be created.")
                if d in self.context_config['forbid_creation_of']:
                    raise strax.DataNotAvailable(
                        f"{d} for {run_id} not found in any storage, and "
                        "your context specifies it cannot be created.")
                to_compute[d] = p
                for dep_d in p.depends_on:
                    check_cache(dep_d)

            # Should we save this data? If not, return.
            if (loading_this_data
                    and not self.context_config['storage_converter']):
                return
            if p.save_when == strax.SaveWhen.NEVER:
                if d in save:
                    raise ValueError(f"Plugin forbids saving of {d}")
                return
            elif p.save_when == strax.SaveWhen.TARGET:
                if d not in targets:
                    return
            elif p.save_when == strax.SaveWhen.EXPLICIT:
                if d not in save:
                    return
            else:
                assert p.save_when == strax.SaveWhen.ALWAYS

            # Warn about conditions that preclude saving, but the user
            # might not expect.
            if time_range is not None:
                # We're not even getting the whole data.
                # Without this check, saving could be attempted if the
                # storage converter mode is enabled.
                self.log.warning(f"Not saving {d} while "
                                 f"selecting a time range in the run")
                return
            if any([len(v) > 0
                    for k, v in self._find_options.items()
                    if 'fuzzy' in k]):
                # In fuzzy matching mode, we cannot (yet) derive the
                # lineage of any data we are creating. To avoid creating
                # false data entries, we currently do not save at all.
                self.log.warning(f"Not saving {d} while fuzzy matching is"
                                 f" turned on.")
                return
            if self.context_config['allow_incomplete']:
                self.log.warning(f"Not saving {d} while loading incomplete"
                                 f" data is allowed.")
                return

            # Save the target and any other outputs of the plugin.
            for d_to_save in set([d] + list(p.provides)):
                if d_to_save in savers and len(savers[d_to_save]):
                    # This multi-output plugin was scanned before
                    # let's not create doubled savers
                    assert p.multi_output
                    continue

                key = strax.DataKey(run_id, d_to_save, p.lineage)

                for sf in self.storage:
                    if sf.readonly:
                        continue
                    if loading_this_data:
                        # Usually, we don't save if we're loading
                        if not self.context_config['storage_converter']:
                            continue
                        # ... but in storage converter mode we do:
                        try:
                            sf.find(key,
                                    **self._find_options)
                            # Already have this data in this backend
                            continue
                        except strax.DataNotAvailable:
                            # Don't have it, so let's save it!
                            pass
                    # If we get here, we must try to save
                    try:
                        savers[d_to_save].append(sf.saver(
                            key,
                            metadata=p.metadata(
                                run_id,
                                d_to_save)))
                    except strax.DataNotAvailable:
                        # This frontend cannot save. Too bad.
                        pass

        for d in targets:
            check_cache(d)
        plugins = to_compute

        intersec = list(plugins.keys() & loaders.keys())
        if len(intersec):
            raise RuntimeError(f"{intersec} both computed and loaded?!")

        # For the plugins which will run computations,
        # check all required options are available or set defaults.
        # Also run any user-defined setup
        for d in plugins.values():
            self._set_plugin_config(d, run_id, tolerant=False)
            d.setup()
        return strax.ProcessorComponents(
            plugins=plugins,
            loaders=loaders,
            savers=dict(savers),
            targets=targets)

    def estimate_run_start(self, run_id, targets=None):
        """Return run start time in ns since epoch.

        This fetches from run metadata, and if this fails, it
        estimates it using data metadata from targets.
        """
        try:
            # Use run metadata, if it is available, to get
            # the run start time (floored to seconds)
            t0 = self.run_metadata(run_id, 'start')['start']
            t0 = t0.replace(tzinfo=datetime.timezone.utc)
            return int(t0.timestamp()) * int(1e9)
        except (strax.RunMetadataNotAvailable, KeyError):
            pass
        # Get an approx start from the data itself,
        # then floor it to seconds for consistency
        if targets:
            for t in strax.to_str_tuple(targets):
                try:
                    t0 = self.get_meta(run_id, t)['chunks'][0]['start']
                    return (int(t0) // int(1e9)) * int(1e9)
                except strax.DataNotAvailable:
                    pass
        warnings.warn(
            "Could not estimate run start time from "
            "run metadata: assuming it is 0",
            UserWarning)
        return 0


    def to_absolute_time_range(self, run_id, targets, time_range=None,
                               seconds_range=None, time_within=None):
        """Return (start, stop) time in ns since unix epoch corresponding
        to time range.

        :param time_range: (start, stop) time in ns since unix epoch.
        Will be returned without modification
        :param targets: data types. Used only if run metadata is unavailable,
        so run start time has to be estimated from data.
        :param seconds_range: (start, stop) seconds since start of run
        :param time_within: row of strax data (e.g. eent)
        """
        if ((time_range is None)
                + (seconds_range is None)
                + (time_within is None)
                < 2):
            raise RuntimeError("Pass no more than one one of"
                               " time_range, seconds_range, ot time_within")
        if seconds_range is not None:
            t0 = self.estimate_run_start(run_id, targets)
            time_range = (t0 + int(1e9 * seconds_range[0]),
                          t0 + int(1e9 * seconds_range[1]))
        if time_within is not None:
            time_range = (time_within['time'], strax.endtime(time_within))
        if time_range is not None:
            # Force time range to be integers, since float math on large numbers
            # in not precise
            time_range = tuple([int(x) for x in time_range])
        return time_range

    def get_iter(self, run_id: str,
                 targets, save=tuple(), max_workers=None,
                 time_range=None,
                 seconds_range=None,
                 time_within=None,
                 time_selection='fully_contained',
                 selection_str=None,
                 keep_columns=None,
                 _chunk_number=None,
                 progress_bar=True,
                 **kwargs) -> ty.Iterator[strax.Chunk]:
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

        # Convert alternate time arguments to absolute range
        time_range = self.to_absolute_time_range(
            run_id=run_id, targets=targets,
            time_range=time_range, seconds_range=seconds_range,
            time_within=time_within)

        # If multiple targets of the same kind, create a MergeOnlyPlugin
        # to merge the results automatically
        if isinstance(targets, (list, tuple)) and len(targets) > 1:
            plugins = self._get_plugins(targets=targets, run_id=run_id)
            if len(set(plugins[d].data_kind_for(d) for d in targets)) == 1:
                temp_name = ('_temp_'
                             + ''.join(
                               random.choices(string.ascii_lowercase, k=10)))
                p = type(temp_name,
                         (strax.MergeOnlyPlugin,),
                         dict(depends_on=tuple(targets)))
                self.register(p)
                targets = (temp_name,)
            else:
                raise RuntimeError("Cannot automerge different data kinds!")

        components = self.get_components(run_id,
                                         targets=targets,
                                         save=save,
                                         time_range=time_range,
                                         chunk_number=_chunk_number)

        # Cleanup the temp plugins
        for k in list(self._plugin_class_registry.keys()):
            if k.startswith('_temp'):
                del self._plugin_class_registry[k]

        seen_a_chunk = False
        generator = strax.ThreadedMailboxProcessor(
                components,
                max_workers=max_workers,
                allow_shm=self.context_config['allow_shm'],
                allow_multiprocess=self.context_config['allow_multiprocess'],
                allow_rechunk=self.context_config['allow_rechunk'],
                allow_lazy=self.context_config['allow_lazy'],
                max_messages=self.context_config['max_messages'],
                timeout=self.context_config['timeout']).iter()

        if progress_bar:
            # Defining time ranges for the progress bar:
            if time_range:
                # user specified a time selection
                start_time, end_time = time_range
            else:
                # If no selection is specified we have to get the last end_time:
                start_time = 0 
                end_time = float('inf')
                for t in strax.to_str_tuple(targets):
                    try:
                        # Sometimes some metadata might be missing e.g. during tests.
                        chunks = self.get_meta(run_id, t)['chunks']  
                        start_time = max(start_time, chunks[0]['start'])
                        end_time = min(end_time, chunks[-1]['end'])
                    except (strax.DataNotAvailable, KeyError):
                        # Maybe at least one target had some metadata.
                        start_time = max(start_time, 0)
                        end_time = min(end_time, float('inf'))

            # Define nice progressbar format:
            bar_format = "{desc}: |{bar}| {percentage:.2f} % [{elapsed}<{remaining}],"\
                         " {postfix[0]} {postfix[1][spc]:.2f} s/chunk,"\
                         " #chunks processed: {postfix[1][n]}"
            sec_per_chunk = np.nan  # Have not computed any chunk yet.
            post_fix = ['Rate last Chunk:', {'spc': sec_per_chunk, 'n': 0}]

        try:
            with contextlib.ExitStack() as stack:
                if start_time != 0 and end_time != float('inf'):
                    # Get initial time
                    pbar = stack.enter_context(tqdm(total=1, postfix=post_fix, bar_format=bar_format))
                    last_time = pbar.last_print_t
                else:
                    progress_bar = False
                
                for n_chunks, result in enumerate(strax.continuity_check(generator), 1):
                    seen_a_chunk = True
                    if not isinstance(result, strax.Chunk):
                        raise ValueError(f"Got type {type(result)} rather than "
                                         f"a strax Chunk from the processor!")
                    result.data = self.apply_selection(
                        result.data,
                        selection_str=selection_str,
                        keep_columns=keep_columns,
                        time_range=time_range,
                        time_selection=time_selection)
                    
                    if progress_bar:
                        # Update progressbar:
                        pbar.n = (result.end - start_time) / (end_time - start_time)
                        pbar.update(0)
                        # Now get last time printed and refresh seconds_per_chunk:
                        # This is a small work around since we do not know the
                        # pacemaker here and therefore we do not know the number of
                        # chunks.
                        sec_per_chunk = pbar.last_print_t - last_time
                        pbar.postfix[1]['spc'] = sec_per_chunk
                        pbar.postfix[1]['n'] = n_chunks
                        pbar.refresh()
                        last_time = pbar.last_print_t

                    yield result

        except GeneratorExit:
            generator.throw(OutsideException(
                "Terminating due to an exception originating from outside "
                "strax's get_iter (which we cannot retrieve)."))

        except Exception as e:
            generator.throw(e)
            raise

        if not seen_a_chunk:
            if time_range is None:
                raise strax.DataCorrupted("No data returned!")
            raise ValueError(f"Invalid time range: {time_range}, "
                             "returned no chunks!")

    def apply_selection(self, x,
                        selection_str=None,
                        keep_columns=None,
                        time_range=None,
                        time_selection='fully_contained'):
        """Return x after applying selections

        :param x: Numpy structured array
        :param selection_str: Query string or sequence of strings to apply.
        :param time_range: (start, stop) range to load, in ns since the epoch
        :param time_selection: Kind of time selectoin to apply:
        - skip: Do not select a time range, even if other arguments say so
        - touching: select things that (partially) overlap with the range
        - fully_contained: (default) select things fully contained in the range

        The right bound is, as always in strax, considered exclusive.
        Thus, data that ends (exclusively) exactly at the right edge of a
        fully_contained selection is returned.
        """
        # Apply the time selections
        if time_range is None or time_selection == 'skip':
            pass
        elif time_selection == 'fully_contained':
            x = x[(time_range[0] <= x['time']) &
                  (strax.endtime(x) <= time_range[1])]
        elif time_selection == 'touching':
            x = x[(strax.endtime(x) > time_range[0]) &
                  (x['time'] < time_range[1])]
        else:
            raise ValueError(f"Unknown time_selection {time_selection}")

        if selection_str:
            if isinstance(selection_str, (list, tuple)):
                selection_str = ' & '.join(f'({x})' for x in selection_str)

            mask = numexpr.evaluate(selection_str, local_dict={
                fn: x[fn]
                for fn in x.dtype.names})
            x = x[mask]

        if keep_columns:
            keep_columns = strax.to_str_tuple(keep_columns)

            # Construct the new dtype
            new_dtype = []
            for unpacked_dtype in strax.unpack_dtype(x.dtype):
                field_name = unpacked_dtype[0]
                if isinstance(field_name, tuple):
                    field_name = field_name[1]
                if field_name in keep_columns:
                    new_dtype.append(unpacked_dtype)

            # Copy over the data
            x2 = np.zeros(len(x), dtype=new_dtype)
            for field_name in keep_columns:
                x2[field_name] = x[field_name]
            x = x2
            del x2

        return x

    def make(self, run_id: ty.Union[str, tuple, list],
             targets, save=tuple(), max_workers=None,
             _skip_if_built=True,
             **kwargs) -> None:
        """Compute target for run_id. Returns nothing (None).
        {get_docs}
        """
        # Multi-run support
        run_ids = strax.to_str_tuple(run_id)
        if len(run_ids) == 0:
            raise ValueError("Cannot build empty list of runs")
        if len(run_ids) > 1:
            return strax.multi_run(
                self.get_array, run_ids, targets=targets,
                throw_away_result=True,
                save=save, max_workers=max_workers, **kwargs)

        if _skip_if_built and self.is_stored(run_id, targets):
            return

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
            results = strax.multi_run(
                self.get_array, run_ids, targets=targets,
                save=save, max_workers=max_workers, **kwargs)
        else:
            source = self.get_iter(
                run_ids[0],
                targets,
                save=save,
                max_workers=max_workers,
                **kwargs)
            results = [x.data for x in source]
        return np.concatenate(results)

    def accumulate(self,
                   run_id: ty.Union[str, tuple, list],
                   targets,
                   fields=None,
                   function=None,
                   store_first_for_others=True,
                   function_takes_fields=False,
                   **kwargs):
        """Return a dictionary with the sum of the result of get_array.

        :param function: Apply this function to the array before summing the
            results. Will be called as function(array), where array is
            a chunk of the get_array result.
            Should return either:
               * A scalar or 1d array -> accumulated result saved under 'result'
               * A record array or dict -> fields accumulated individually
               * None -> nothing accumulated
            If not provided, the identify function is used.

        :param fields: Fields of the function output to accumulate.
            If not provided, all output fields will be accumulated.

        :param store_first_for_others: if True (default), for fields included
            in the data but not fields, store the first value seen in the data
            (if any value is seen).

        :param function_takes_fields: If True, function will be called as
            function(data, fields) instead of function(data).

        All other options are as for get_iter.

        :returns dictionary: Dictionary with the accumulated result;
            see function and store_first_for_others arguments.
            Four fields are always added:
                start: start time of the first processed chunk
                end: end time of the last processed chunk
                n_chunks: number of chunks in run
                n_rows: number of data entries in run
        """
        n_chunks = 0
        seen_data = False
        result = {'n_rows': 0}
        if fields is not None:
            fields = strax.to_str_tuple(fields)
        if function is None:
            def function(arr):
                return arr
            function_takes_fields = False

        for chunk in self.get_iter(run_id, targets, **kwargs):
            data = chunk.data

            if n_chunks == 0:
                result['start'] = chunk.start
                if fields is None:
                    # Sum all fields except time and endtime
                    fields = [x for x in data.dtype.names
                              if x not in ('time', 'endtime')]

            if store_first_for_others and not seen_data and len(data):
                # Store the first value we see for the non-accumulated fields
                for name in data.dtype.names:
                    if name in fields:
                        result[name] = data[0][name]
                seen_data = True
            result['end'] = chunk.end
            result['n_rows'] += len(data)

            # Run the function
            if function_takes_fields:
                data = function(data, fields)
            else:
                data = function(data)

            # Accumulate the result
            # Don't try to be clever here,
            #   += doesn't work on readonly array fields;
            #   .sum() doesn't work on scalars
            if data is None:
                pass

            elif (isinstance(data, dict)
                  or (isinstance(data, np.ndarray)
                      and data.dtype.fields is not None)):
                # Function returned record array or dict
                for field in fields:
                    result[field] = (
                        result.get(field, 0)
                        + np.sum(data[field], axis=0))
            else:
                # Function returned a scalar or flat array
                result['result'] = (
                        np.sum(data, axis=0)
                        + result.get('result', 0))

            n_chunks += 1

        result['n_chunks'] = n_chunks
        return result

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

    def key_for(self, run_id, target):
        """Get the DataKey for a given run and a given target plugin. The
        DataKey is inferred from the plugin lineage.

        :param run_id: run id to get
        :param target: data type to get
        :return: strax.DataKey of the target
        """
        p = self._get_plugins((target,), run_id)[target]
        return strax.DataKey(run_id, target, p.lineage)

    def get_meta(self, run_id, target) -> dict:
        """Return metadata for target for run_id, or raise DataNotAvailable
        if data is not yet available.

        :param run_id: run id to get
        :param target: data type to get
        """
        key = self.key_for(run_id, target)
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
        raise strax.RunMetadataNotAvailable(f"No run-level metadata available "
                                            f"for {run_id}")

    def size_mb(self, run_id, target):
        """Return megabytes of memory required to hold data"""
        md = self.get_meta(run_id, target)
        return sum([x['nbytes'] for x in md['chunks']]) / 1e6

    def run_defaults(self, run_id):
        """Get configuration defaults from the run metadata (if these exist)

        This will only call the rundb once for each run while the context is
        in existence; further calls to this will return a cached value.
        """
        if not self.context_config['use_per_run_defaults']:
            return dict()
        if run_id in self._run_defaults_cache:
            return self._run_defaults_cache[run_id]
        try:
            defs = self.run_metadata(
                run_id,
                projection=RUN_DEFAULTS_KEY).get(RUN_DEFAULTS_KEY, dict())
        except strax.RunMetadataNotAvailable:
            defs = dict()
        self._run_defaults_cache[run_id] = defs
        return defs

    def is_stored(self, run_id, target, **kwargs):
        """Return whether data type target has been saved for run_id
        through any of the registered storage frontends.

        Note that even if False is returned, the data type may still be made
        with a trivial computation.
        """
        if isinstance(target, (tuple, list)):
            return all([self.is_stored(run_id, t, **kwargs)
                        for t in target])

        # If any new options given, replace the current context
        # with a temporary one
        # TODO duplicated code with with get_iter
        if len(kwargs):
            # Comment below disables pycharm from inspecting the line below it
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        key = self.key_for(run_id, target)
        for sf in self.storage:
            try:
                sf.find(key, **self._find_options)
                return True
            except strax.DataNotAvailable:
                continue
        return False

    def _check_forbidden(self):
        """Check that the forbid_creation_of config is of tuple type.
        Otherwise, try to make it a tuple"""
        self.context_config['forbid_creation_of'] = strax.to_str_tuple(
            self.context_config['forbid_creation_of'])

    @classmethod
    def add_method(cls, f):
        """Add f as a new Context method"""
        setattr(cls, f.__name__, f)


select_docs = """
:param selection_str: Query string or sequence of strings to apply.
:param time_range: (start, stop) range to load, in ns since the epoch
:param seconds_range: (start, stop) range of seconds since
the start of the run to load.
:param time_within: row of strax data (e.g. event) to use as time range
:param time_selection: Kind of time selectoin to apply:
- fully_contained: (default) select things fully contained in the range
- touching: select things that (partially) overlap with the range
- skip: Do not select a time range, even if other arguments say so
:param _chunk_number: For internal use: return data from one chunk.
:param progress_bar: Display a progress bar if metedata exists.
"""

get_docs = """
:param run_id: run id to get
:param targets: list/tuple of strings of data type names to get
:param save: extra data types you would like to save
    to cache, if they occur in intermediate computations.
    Many plugins save automatically anyway.
:param max_workers: Number of worker threads/processes to spawn.
    In practice more CPUs may be used due to strax's multithreading.
""" + select_docs


for attr in dir(Context):
    attr_val = getattr(Context, attr)
    if hasattr(attr_val, '__doc__'):
        doc = attr_val.__doc__
        if doc is not None and '{get_docs}' in doc:
            attr_val.__doc__ = doc.format(get_docs=get_docs)


@export
class OutsideException(Exception):
    pass
