import datetime
import logging
import warnings
import fnmatch
from functools import partial
import typing as ty
import time
import numpy as np
import pandas as pd
import strax
import inspect
import types
from collections import defaultdict
from immutabledict import immutabledict
from enum import IntEnum


export, __all__ = strax.exporter()
__all__ += ['RUN_DEFAULTS_KEY']

RUN_DEFAULTS_KEY = 'strax_defaults'

# use tqdm as loaded in utils (from tqdm.notebook when in a juypyter env)
tqdm = strax.utils.tqdm


@strax.takes_config(
    strax.Option(name='storage_converter', default=False, type=bool,
                 help='If True, save data that is loaded from one frontend '
                      'through all willing other storage frontends.'),
    strax.Option(name='fuzzy_for', default=tuple(), type=tuple,
                 help='Tuple or string of plugin names for which no checks for version, '
                      'providing plugin, and config will be performed when '
                      'looking for data.'),
    strax.Option(name='fuzzy_for_options', default=tuple(), type=tuple,
                 help='Tuple of config options for which no checks will be '
                      'performed when looking for data.'),
    strax.Option(name='allow_incomplete', default=False, type=bool,
                 help="Allow loading of incompletely written data, if the "
                      "storage systems support it"),
    strax.Option(name='allow_rechunk', default=True, type=bool,
                 help="Allow rechunking of data during writing."),
    strax.Option(name='allow_multiprocess', default=False, type=bool,
                 help="Allow multiprocessing."
                      "If False, will use multithreading only."),
    strax.Option(name='allow_shm', default=False, type=bool,
                 help="Allow use of /dev/shm for interprocess communication."),
    strax.Option(name='allow_lazy', default=True, type=bool,
                 help='Allow "lazy" processing. Saves memory, but incompatible '
                      'with multiprocessing and perhaps slightly slower.'),
    strax.Option(name='forbid_creation_of', default=tuple(), type=tuple,
                 help="If any of the following datatypes is requested to be "
                      "created, throw an error instead. Useful to limit "
                      "descending too far into the dependency graph."),
    strax.Option(name='store_run_fields', default=tuple(), type=tuple,
                 help="Tuple of run document fields to store "
                      "during scan_run."),
    strax.Option(name='check_available', default=tuple(), type=tuple,
                 help="Tuple of data types to scan availability for "
                      "during scan_run."),
    strax.Option(name='max_messages', default=4, type=int,
                 help="Maximum number of mailbox messages, i.e. size of buffer "
                      "between plugins. Too high = RAM blows up. "
                      "Too low = likely deadlocks."),
    strax.Option(name='timeout', default=24 * 3600, type=int,
                 help="Terminate processing if any one mailbox receives "
                      "no result for more than this many seconds"),
    strax.Option(name='saver_timeout', default=900, type=int,
                 help="Max time [s] a saver can take to store a result. Set "
                      "high for slow compression algorithms."),
    strax.Option(name='use_per_run_defaults', default=False, type=bool,
                 help='Scan the run db for per-run defaults. '
                      'This is an experimental strax feature that will '
                      'possibly be removed, see issue #246'),
    strax.Option(name='free_options', default=tuple(), type=(tuple,list),
                 help='Do not warn if any of these options are passed, '
                      'even when no registered plugin takes them.'),
    strax.Option(name='apply_data_function', default=tuple(),
                 type=(tuple, list, ty.Callable),
                 help='Apply a function to the data prior to returning the'
                      'data. The function should take three positional arguments: '
                      'func(<data>, <run_id>, <targets>).'),
    strax.Option(name='write_superruns', default=False, type=bool,
                 help='If True, save superruns as rechunked "new" data.'),
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
    _fixed_plugin_cache: dict = None
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
                self.log.warning(f"Unknown config option {k}; will do nothing.")

        self.context_config = new_config

        for k in self.context_config:
            if k not in self.takes_config:
                self.log.warning(f"Invalid context option {k}; will do nothing.")

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
                self._per_run_default_allowed_check(option, items)
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

    def deregister_plugins_with_missing_dependencies(self):
        """
        Deregister plugins in case a data_type the plugin
        depends on is not provided by any other plugin.
        """
        registry_changed = True
        while registry_changed:
            all_provides = set()
            plugins_to_deregister = []

            for p in self._plugin_class_registry.values():
                all_provides |= set(p.provides)

            for p_key, p in self._plugin_class_registry.items():
                requires = set(strax.to_str_tuple(p.depends_on))
                if not requires.issubset(all_provides):
                    plugins_to_deregister.append(p_key)

            for p_key in plugins_to_deregister:
                self.log.info(f'Deregister {p_key}')
                del self._plugin_class_registry[p_key]

            if not len(plugins_to_deregister):
                registry_changed = False

    def search_field(self,
                     pattern: str,
                     include_code_usage: bool = True,
                     return_matches: bool = False,
                     ) -> ty.Union[None, ty.Tuple[defaultdict, dict]]:
        """
        Find and print which plugin(s) provides a field that matches
        pattern (fnmatch).

        :param pattern: pattern to match, e.g. 'time' or 'tim*'
        :param include_code_usage: Also include the code occurrences of
            the fields that match the pattern.
        :param return_matches: If set, return a dictionary with the
            matching fields and the occurrences in code.
        :return: when return_matches is set, return a dictionary with
            the matching fields and the occurrences in code. Otherwise,
            we are not returning anything and just print the results
        """
        cache = dict()
        field_matches = defaultdict(list)
        code_matches = dict()
        for data_type in sorted(list(self._plugin_class_registry.keys())):
            if data_type not in cache:
                cache.update(self._get_plugins((data_type,), run_id='0'))
            plugin = cache[data_type]

            for field_name in plugin.dtype_for(data_type).names:
                if fnmatch.fnmatch(field_name, pattern):
                    field_matches[field_name].append((data_type, plugin.__class__.__name__))
                    if field_name in code_matches:
                        continue
                    # we need to do this for 'field_name' rather than pattern
                    # since we want an exact match (otherwise too fuzzy with
                    # comments etc.) Do this once, for all the plugins.
                    fields_used = self.search_field_usage(field_name, plugin=None)
                    if include_code_usage and fields_used:
                        code_matches[field_name] = fields_used
        if return_matches:
            return field_matches, code_matches

        # Print the results and return nothing
        for field_name, matches in field_matches.items():
            print()
            for data_type, name in matches:
                print(f"{field_name} is part of {data_type} (provided by {name})")
        for field_name, functions in code_matches.items():
            print()
            for function in functions:
                print(f"{field_name} is used in {function}")

    def search_field_usage(self,
                           search_string: str,
                           plugin: ty.Union[strax.Plugin, ty.List[strax.Plugin], None] = None
                           ) -> ty.List[str]:
        """
        Find and return which plugin(s) use a given field.

        :param search_string: a field that matches pattern exact
        :param plugin: plugin where to look for a field
        :return: list of code occurrences in the form of PLUGIN.FUNCTION
        """
        if plugin is None:
            plugin = list(self._plugin_class_registry.values())
        if not isinstance(plugin, (list, tuple)):
            plugin = [plugin]
        result = []
        for plug in plugin:
            for attribute_name, class_attribute in plug.__dict__.items():
                is_function = isinstance(class_attribute, types.FunctionType)
                if is_function:
                    for line in inspect.getsource(class_attribute).split('\n'):
                        if search_string in line:
                            if plug.__class__.__name__ == 'type':
                                # Make sure we have the instance, not the class:
                                # >>> class A: pass
                                # >>> A.__class__.__name__
                                # 'type'
                                # >>> A().__class__.__name__
                                # 'A'
                                plug = plug()
                            result += [f'{plug.__class__.__name__}.{attribute_name}']
                            # Likely to be used several other times
                            break
        return result

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
            if not isinstance(x, type(type)):
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

        if p.child_plugin:
            # This plugin is a child of another plugin. This means we have to overwrite
            # the registered option settings in p.config with the options specified by the
            # child. This is required since the super().compute() method in a child plugins
            # will still point to the option names of the parent (e.g. self.config['parent_name']).

            # options to pass. So update parent config according to child:
            for option_name, opt in p.takes_config.items():
                # Now loop again overall options for this plugin (parent + child)
                # and get all child options:
                if opt.child_option:
                    # See if option is tagged as a child option. In that case replace the
                    # config value of the parent with the value of the child
                    option_value = config[option_name]
                    parent_name = opt.parent_option_name

                    mes = (f'Cannot find "{parent_name}" among the options of the parent.'
                           f' Either you specified by accident {option_name} as child option'
                           f' or you specified the wrong parent_option_name. Have you specified '
                           'the correct parent option name?')
                    assert parent_name in p.config, mes
                    p.config[parent_name] = option_value

    def _context_hash(self):
        """
        Dump the current config + plugin class registry to a hash as a
        sanity check for building the _fixed_plugin_cache. If any item
        changes in the config, so does this hash.
        """
        base_hash_on_config = self.config.copy()
        # Also take into account the versions of the plugins registered
        base_hash_on_config.update(
            {data_type: (plugin.__version__, plugin.compressor, plugin.input_timeout)
             for data_type, plugin in self._plugin_class_registry.items()
             if not data_type.startswith('_temp_')
             })
        return strax.deterministic_hash(base_hash_on_config)

    def _plugins_are_cached(self, targets: ty.Tuple[str],) -> bool:
        """Check if all the requested targets are in the _fixed_plugin_cache"""
        if self.context_config['use_per_run_defaults'] or self._fixed_plugin_cache is None:
            # There is no point in caching if plugins (lineage) can
            # change per run or the cache is empty.
            return False

        context_hash = self._context_hash()
        if context_hash not in self._fixed_plugin_cache:
            return False
        plugin_cache = self._fixed_plugin_cache[context_hash]
        return all([t in plugin_cache for t in targets])

    def _plugins_to_cache(self, plugins: dict) -> None:
        if self.context_config['use_per_run_defaults']:
            # There is no point in caching if plugins (lineage) can change per run
            return
        context_hash = self._context_hash()
        if self._fixed_plugin_cache is None:
            self._fixed_plugin_cache = {context_hash: dict()}
        elif context_hash not in self._fixed_plugin_cache:
            # Create a new cache every time the hash is not matching to
            # save memory. If a config changes, building the cache again
            # should be fast, we just need to track which cache to use.
            self.log.info('Replacing context._fixed_plugin_cache since '
                          'plugins/versions changed')
            self._fixed_plugin_cache = {context_hash: dict()}
        for target, plugin in plugins.items():
            self._fixed_plugin_cache[context_hash][target] = plugin

    def _fix_dependency(self, plugin_registry: dict, end_plugin: str):
        """
        Starting from end-plugin, fix the dtype until there is nothing
        left to fix. Keep in mind that dtypes can be chained.
        """
        for go_to in plugin_registry[end_plugin].depends_on:
            self._fix_dependency(plugin_registry, go_to)
        plugin_registry[end_plugin].fix_dtype()

    def __get_plugins_from_cache(self,
                                 run_id: str) -> ty.Dict[str, strax.Plugin]:
        # Doubly underscored since we don't do any key-checks etc here
        """Load requested plugins from the plugin_cache"""
        requested_plugins = {}
        for target, plugin in self._fixed_plugin_cache[self._context_hash()].items():
            if target in requested_plugins:
                # If e.g. target is already seen because the plugin is
                # multi output
                continue

            requested_p = plugin.__copy__()
            requested_p.run_id = run_id

            # Re-use only one instance if the plugin is multi output
            for provides in strax.to_str_tuple(requested_p.provides):
                requested_plugins[provides] = requested_p

        # At this stage, all the plugins should be in requested_plugins
        # To prevent infinite copying, we are only now linking the
        # dependencies of each plugin to another where needed.
        for target, plugin in requested_plugins.items():
            plugin.deps = {dependency: requested_plugins[dependency]
                           for dependency in plugin.depends_on
                           }
        # Finally, fix the dtype. Since infer_dtype may depend on the
        # entire deps chain, we need to start at the last plugin and go
        # all the way down to the lowest level.
        for final_plugins in self._get_end_targets(requested_plugins):
            self._fix_dependency(requested_plugins, final_plugins)
        return requested_plugins

    def _get_plugins(self,
                     targets: ty.Tuple[str],
                     run_id: str) -> ty.Dict[str, strax.Plugin]:
        """Return dictionary of plugin instances necessary to compute targets
        from scratch.
        For a plugin that produces multiple outputs, we make only a single
        instance, which is referenced under multiple keys in the output dict.
        """
        if self._plugins_are_cached(targets):
            return self.__get_plugins_from_cache(run_id)

        # Check all config options are taken by some registered plugin class
        # (helps spot typos)
        all_opts = set().union(*[
            pc.takes_config.keys()
            for pc in self._plugin_class_registry.values()])
        for k in self.config:
            if not (k in all_opts or k in self.context_config['free_options']):
                self.log.warning(f"Option {k} not taken by any registered plugin")

        # Initialize plugins for the entire computation graph
        # (most likely far further down than we need)
        # to get lineages and dependency info.
        def get_plugin(data_type):
            nonlocal non_local_plugins

            if data_type not in self._plugin_class_registry:
                raise KeyError(f"No plugin class registered that provides {data_type}")

            plugin = self._plugin_class_registry[data_type]()

            d_provides = None  # just to make codefactor happy
            for d_provides in plugin.provides:
                non_local_plugins[d_provides] = plugin

            plugin.run_id = run_id

            # The plugin may not get all the required options here
            # but we don't know if we need the plugin yet
            self._set_plugin_config(plugin, run_id, tolerant=True)

            plugin.deps = {d_depends: get_plugin(d_depends) for d_depends in plugin.depends_on}

            last_provide = d_provides

            if plugin.child_plugin:
                # Plugin is a child of another plugin, hence we have to
                # drop the parents config from the lineage
                configs = {}

                # Getting information about the parent:
                parent_class = plugin.__class__.__bases__[0]
                # Get all parent options which are overwritten by a child:
                parent_options = [option.parent_option_name for option in plugin.takes_config.values()
                                  if option.child_option]

                for option_name, v in plugin.config.items():
                    # Looping over all settings, option_name is either the option name of the
                    # parent or the child.
                    if option_name in parent_options:
                        # In case it is the parent we continue
                        continue

                    if plugin.takes_config[option_name].track:
                        # Add all options which should be tracked:
                        configs[option_name] = v

                # Also adding name and version of the parent to the lineage:
                configs[parent_class.__name__] = parent_class.__version__

                plugin.lineage = {last_provide: (
                    plugin.__class__.__name__,
                    plugin.version(run_id),
                    configs)}
            else:
                plugin.lineage = {last_provide: (
                    plugin.__class__.__name__,
                    plugin.version(run_id),
                    {option: setting for option, setting
                     in plugin.config.items()
                     if plugin.takes_config[option].track})}
            for d_depends in plugin.depends_on:
                plugin.lineage.update(plugin.deps[d_depends].lineage)

            if not hasattr(plugin, 'data_kind') and not plugin.multi_output:
                if len(plugin.depends_on):
                    # Assume data kind is the same as the first dependency
                    first_dep = plugin.depends_on[0]
                    plugin.data_kind = plugin.deps[first_dep].data_kind_for(first_dep)
                else:
                    # No dependencies: assume provided data kind and
                    # data type are synonymous
                    plugin.data_kind = plugin.provides[0]

            plugin.fix_dtype()

            return plugin

        non_local_plugins = {}
        for t in targets:
            p = get_plugin(t)
            non_local_plugins[t] = p

        self._plugins_to_cache(non_local_plugins)
        return non_local_plugins

    def _per_run_default_allowed_check(self, option_name, option):
        """Check if an option of a registered plugin is allowed"""
        per_run_default = option.default_by_run != strax.OMITTED
        not_overwritten = option_name not in self.config
        per_run_is_forbidden = not self.context_config['use_per_run_defaults']
        if per_run_default and not_overwritten and per_run_is_forbidden:
            raise strax.InvalidConfiguration(
                f'{option_name} is specified as a per-run-default which is not '
                f'allowed by the context')

    @staticmethod
    def _get_end_targets(plugins: dict) -> ty.Tuple[str]:
        """
        Get the datatype that is provided by a plugin but not depended
            on by any other plugin
        """
        provides = [prov for p in plugins.values()
                    for prov in strax.to_str_tuple(p.provides)]
        depends_on = [dep for p in plugins.values()
                      for dep in strax.to_str_tuple(p.depends_on)]
        uniques = list(set(provides) ^ set(depends_on))
        return strax.to_str_tuple(uniques)

    @property
    def _find_options(self):

        # The plugin settings in the lineage are stored with the last
        # plugin provides name as a key. This can be quite confusing
        # since e.g. to be fuzzy for the peaklets settings the user has
        # to specify fuzzy_for=('lone_hits'). Here a small work around
        # to change this and not to reprocess the entire data set.
        fuzzy_for_keys = strax.to_str_tuple(self.context_config['fuzzy_for'])
        last_provides = []
        for key in fuzzy_for_keys:
            last_provides.append(self._plugin_class_registry[key].provides[-1])
        last_provides = tuple(last_provides)

        return dict(fuzzy_for=last_provides,
                    fuzzy_for_options=self.context_config['fuzzy_for_options'],
                    allow_incomplete=self.context_config['allow_incomplete'])

    @property
    def _sorted_storage(self) -> ty.List[strax.StorageFrontend]:
        """
        Simple ordering of the storage frontends on the fly when e.g.
        looking for data. This allows us to use the simple self.storage
        as a simple list without asking users to keep any particular
        order in mind. Return the fastest first and try loading from it
        """
        return sorted(self.storage, key=lambda x: x.storage_type)

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
        for sf in self._sorted_storage:
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
                       time_range=None, chunk_number=None,
                       ) -> strax.ProcessorComponents:
        """Return components for setting up a processor
        {get_docs}
        """
        save = strax.to_str_tuple(save)
        targets = strax.to_str_tuple(targets)

        for t in targets:
            if len(t) == 1:
                raise ValueError(f"Plugin names must be more than one letter, not {t}")

        plugins = self._get_plugins(targets, run_id)

        # Get savers/loaders, and meanwhile filter out plugins that do not
        # have to do computation. (their instances will stick around
        # though the .deps attribute of plugins that do)
        loaders = dict()
        loader_plugins = dict()
        savers = dict()
        seen = set()
        to_compute = dict()

        def check_cache(target_i):
            """For some target, add loaders, and savers where appropriate"""
            nonlocal plugins, loaders, savers, seen
            if target_i in seen:
                return
            seen.add(target_i)
            target_plugin = plugins[target_i]

            # Can we load this data?
            loading_this_data = False
            key = self.key_for(run_id, target_i)

            loader = self._get_partial_loader_for(
                key,
                chunk_number=chunk_number,
                time_range=time_range)

            _is_superrun = (run_id.startswith('_') and
                            not target_plugin.provides[0].startswith('_temp'))
            if not loader and _is_superrun:
                if time_range is not None:
                    raise NotImplementedError("time range loading not yet "
                                              "supported for superruns")

                sub_run_spec = self.run_metadata(
                    run_id, 'sub_run_spec')['sub_run_spec']

                # Make subruns if they do not exist.
                self.make(list(sub_run_spec.keys()), target_i, save=(target_i,))

                ldrs = []
                for subrun in sub_run_spec:
                    sub_key = self.key_for(subrun, target_i)

                    if sub_run_spec[subrun] == 'all':
                        _subrun_time_range = None
                    else:
                        _subrun_time_range = sub_run_spec[subrun]
                    loader = self._get_partial_loader_for(
                        sub_key,
                        time_range=_subrun_time_range,
                        chunk_number=chunk_number)
                    if not loader:
                        raise RuntimeError(
                            f"Could not load {target_i} for subrun {subrun} "
                             "even though we made it? Is the plugin "
                             "you are requesting a SaveWhen.NEVER-plguin?")
                    ldrs.append(loader)

                def concat_loader(*args, **kwargs):
                    for x in ldrs:
                        yield from x(*args, **kwargs)
                # pylint: disable=unnecessary-lambda
                loader = lambda *args, **kwargs: concat_loader(*args, **kwargs)

            if loader:
                # Found it! No need to make it or look in other frontends
                loading_this_data = True
                loaders[target_i] = loader
                loader_plugins[target_i] = target_plugin
                del plugins[target_i]
            else:
                # Data not found anywhere. We will be computing it.
                self._check_forbidden()
                if (time_range is not None
                        and target_plugin.save_when[target_i] > strax.SaveWhen.EXPLICIT):
                    # While the data type providing the time information is
                    # available (else we'd have failed earlier), one of the
                    # other requested data types is not.
                    error_message = (
                        f"Time range selection assumes data is already available,"
                        f" but {target_i} for {run_id} is not.")
                    if target_plugin.save_when[target_i] == strax.SaveWhen.TARGET:
                        error_message += (f"\nFirst run st.make({run_id}, "
                                          f"{target_i}) to make {target_i}.")
                    raise strax.DataNotAvailable(error_message)
                if '*' in self.context_config['forbid_creation_of']:
                    raise strax.DataNotAvailable(
                        f"{target_i} for {run_id} not found in any storage, and "
                        "your context specifies no new data can be created.")
                if target_i in self.context_config['forbid_creation_of']:
                    raise strax.DataNotAvailable(
                        f"{target_i} for {run_id} not found in any storage, and "
                        "your context specifies it cannot be created.")

                to_compute[target_i] = target_plugin
                for dep_d in target_plugin.depends_on:
                    check_cache(dep_d)
            
            if self.context_config['storage_converter']:
                warnings.warn('The storage converter mode will be replaced by "copy_to_frontend" soon. '
                              'It will be removed in one of the future releases. Please let us know if '
                              'you are still using the "storage_converter" option.', DeprecationWarning)
            
            # Should we save this data? If not, return.
            _can_store_superrun = (self.context_config['write_superruns'] and _is_superrun)
            # In case we can load the data already we want either use the storage converter
            # or make a new superrun.
            if (loading_this_data
                    and not self.context_config['storage_converter']
                    and not _can_store_superrun):
                return
 
            # Now we should check whether we meet the saving requirements (Explicit, Target etc.)
            if (not self._target_should_be_saved(
                    target_plugin, target_i, targets, save, loader, _is_superrun)
                    and not self.context_config['storage_converter']):
                # In case of the storage converter mode we copy already existing data. So we do not
                # have to check for the saving requirements here.
                return
            
            # Warn about conditions that preclude saving, but the user
            # might not expect.
            if time_range is not None:
                # We're not even getting the whole data.
                # Without this check, saving could be attempted if the
                # storage converter mode is enabled.
                self.log.warning(f"Not saving {target_i} while "
                                 f"selecting a time range in the run")
                return
            if any([len(v) > 0
                    for k, v in self._find_options.items()
                    if 'fuzzy' in k]):
                # In fuzzy matching mode, we cannot (yet) derive the
                # lineage of any data we are creating. To avoid creating
                # false data entries, we currently do not save at all.
                self.log.warning(f"Not saving {target_i} while fuzzy matching is"
                                 f" turned on.")
                return
            if self.context_config['allow_incomplete']:
                self.log.warning(f"Not saving {target_i} while loading incomplete"
                                 f" data is allowed.")
                return
            
            # Save the target and any other outputs of the plugin.
            if _is_superrun:
                # In case of a superrun we are only interested in the specified targets 
                # and not any other stuff provided by the corresponding plugin.
                savers = self._add_saver(savers, target_i, target_plugin,
                                         _is_superrun, loading_this_data)
            else:
                for d_to_save in set([target_i] + list(target_plugin.provides)):
                    key = self.key_for(run_id, d_to_save)
                    loader = self._get_partial_loader_for(key,
                                                          time_range=time_range,
                                                          chunk_number=chunk_number)

                    if ((not self._target_should_be_saved(
                            target_plugin, d_to_save, targets, save, loader, _is_superrun)
                         and not self.context_config['storage_converter'])
                            or savers.get(d_to_save)):
                        # This multi-output plugin was scanned before
                        # let's not create doubled savers or store data_types we do not want to.
                        assert target_plugin.multi_output
                        continue
                    
                    savers = self._add_saver(savers, d_to_save, target_plugin,
                                             _is_superrun, loading_this_data)

        for target_i in targets:
            check_cache(target_i)
        plugins = to_compute

        intersec = list(plugins.keys() & loaders.keys())
        if len(intersec):
            raise RuntimeError(f"{intersec} both computed and loaded?!")
        if len(targets) > 1:
            final_plugin = [t for t in targets if t in self._get_end_targets(plugins)][:1]
            self.log.warning(
                f'Multiple targets detected! This is only suitable for mass '
                f'producing dataypes since only {final_plugin} will be '
                f'subscribed in the mailbox system!')
        else:
            final_plugin = targets
        # For the plugins which will run computations,
        # check all required options are available or set defaults.
        # Also run any user-defined setup
        for d in plugins.values():
            self._set_plugin_config(d, run_id, tolerant=False)
            d.setup()
        return strax.ProcessorComponents(
            plugins=plugins,
            loaders=loaders,
            loader_plugins=loader_plugins,
            savers=savers,
            targets=strax.to_str_tuple(final_plugin))
    
    def _add_saver(self,
                   savers: dict,
                   d_to_save: str,
                   target_plugin: strax.Plugin,
                   _is_superrun: bool,
                   loading_this_data: bool):
        """
        Adds savers to already existing savers. Checks if data_type can
        be stored in any storage frontend.

        :param savers: Dictionary of already existing savers.
        :param d_to_save: String of the data_type to be saved.
        :param target_plugin: Plugin which produces the data_type
        :param _is_superrun: Boolean if run is a superrun
        :param loading_this_data: Boolean if data can be loaded. Required
            for storing during storage_converter mode and writing of
            superruns.
        :return: Updated savers dictionary.
        """
        key = strax.DataKey(target_plugin.run_id, d_to_save, target_plugin.lineage)
        for sf in self._sorted_storage:
            if sf.readonly:
                continue
            if loading_this_data:
                # Usually, we don't save if we're loading
                if (not self.context_config['storage_converter']
                        and (not self.context_config['write_superruns'] and _is_superrun)):
                    continue
                    # ... but in storage converter mode we do,
                    # ... or we want to write a new superrun. This is different from
                    # storage converter mode as we do not want to write the subruns again.
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
                saver = sf.saver(
                    key,
                    metadata=target_plugin.metadata(target_plugin.run_id, d_to_save),
                    saver_timeout=self.context_config['saver_timeout'])
                # Now that we are surely saving, make an entry in savers
                savers.setdefault(d_to_save, [])
                savers[d_to_save].append(saver)
            except strax.DataNotAvailable:
                # This frontend cannot save. Too bad.
                pass
        return savers

    @staticmethod
    def _target_should_be_saved(target_plugin, target, targets, save, loader, _is_superrun):
        """
        Function which checks if a given target should be saved.

        :param target_plugin: Plugin to compute target data_type.
        :param target: Target data_type.
        :param targets: Other targets to be computed.
        :param loader: Partial loader for the corresponding target
        :param save: Targets to be saved.
        :param _is_superrun: Boolean if run is a superrun.
        """
        if target_plugin.save_when[target] == strax.SaveWhen.NEVER:
            if target in save:
                raise ValueError(f"Plugin forbids saving of {target}")
            return False
        elif target_plugin.save_when[target] == strax.SaveWhen.TARGET:
            if target not in targets:
                return False
        elif target_plugin.save_when[target] == strax.SaveWhen.EXPLICIT:
            if target not in save:
                return False
        elif (target_plugin.save_when[target] == strax.SaveWhen.ALWAYS
              and (loader and not _is_superrun)):
            # If an loader for this data_type exists already we do not
            # have to store the data again, except it is a superrun.
            return False
        return True

    def estimate_run_start_and_end(self, run_id, targets=None):
        """Return run start and end time in ns since epoch.

        This fetches from run metadata, and if this fails, it estimates
            it using data metadata from the targets or the underlying
            data-types (if it is stored).
        """
        try:
            res = []
            for i in ('start', 'end'):
                # Use run metadata, if it is available, to get
                # the run start time (floored to seconds)
                t = self.run_metadata(run_id, i)[i]
                t = t.replace(tzinfo=datetime.timezone.utc)
                t = int(t.timestamp()) * int(1e9)
                res.append(t)
            return res
        except (strax.RunMetadataNotAvailable, KeyError) as e:
            self.log.debug(f'Could not infer start/stop due to type {type(e)} {e}')
            pass
        # Get an approx start from the data itself,
        # then floor it to seconds for consistency
        if targets:
            self.log.debug('Infer start/stop from targets')
            for t in self._get_plugins(strax.to_str_tuple(targets),
                                       run_id,
                                       ).keys():
                if not self.is_stored(run_id, t):
                    continue
                self.log.debug(f'Try inferring start/stop from {t}')
                try:
                    t0 = self.get_meta(run_id, t)['chunks'][0]['start']
                    t0 = (int(t0) // int(1e9)) * int(1e9)

                    t1 = self.get_meta(run_id, t)['chunks'][-1]['end']
                    t1 = (int(t1) // int(1e9)) * int(1e9)
                    return t0, t1
                except strax.DataNotAvailable:
                    pass
        self.log.warning(
            "Could not estimate run start and end time from "
            "run metadata: assuming it is 0 and inf")
        return 0, float('inf')

    def to_absolute_time_range(self, run_id, targets=None, time_range=None,
                               seconds_range=None, time_within=None,
                               full_range=None):
        """Return (start, stop) time in ns since unix epoch corresponding
        to time range.

        :param run_id: run id to get
        :param time_range: (start, stop) time in ns since unix epoch.
        Will be returned without modification
        :param targets: data types. Used only if run metadata is unavailable,
        so run start time has to be estimated from data.
        :param seconds_range: (start, stop) seconds since start of run
        :param time_within: row of strax data (e.g. eent)
        :param full_range: If True returns full time_range of the run.
        """

        selection = ((time_range is None) +
                     (seconds_range is None) +
                     (time_within is None) +
                     (full_range is None))
        if selection < 2:
            raise RuntimeError("Pass no more than one one of"
                               " time_range, seconds_range, time_within"
                               ", or full_range")
        if seconds_range is not None:
            t0, _ = self.estimate_run_start_and_end(run_id, targets)
            time_range = (t0 + int(1e9 * seconds_range[0]),
                          t0 + int(1e9 * seconds_range[1]))
        if time_within is not None:
            time_range = (time_within['time'], strax.endtime(time_within))
        if time_range is not None:
            # Force time range to be integers, since float math on large numbers
            # in not precise
            time_range = tuple([int(x) for x in time_range])

        if full_range:
            time_range = self.estimate_run_start_and_end(run_id, targets)
        return time_range

    def get_iter(self, run_id: str,
                 targets, save=tuple(), max_workers=None,
                 time_range=None,
                 seconds_range=None,
                 time_within=None,
                 time_selection='fully_contained',
                 selection_str=None,
                 keep_columns=None,
                 drop_columns=None,
                 allow_multiple=False,
                 progress_bar=True,
                 _chunk_number=None,
                 **kwargs) -> ty.Iterator[strax.Chunk]:
        """Compute target for run_id and iterate over results.

        Do NOT interrupt the iterator (i.e. break): it will keep running stuff
        in background threads...
        {get_docs}
        """
        if hasattr(run_id, 'decode'):
            # Byte string has to be decoded:
            run_id = run_id.decode('utf-8')

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

        # Keep a copy of the list of targets for apply_function
        # (otherwise potentially overwritten in temp-plugin)
        targets_list = targets

        _is_superrun = run_id.startswith('_')

        # If multiple targets of the same kind, create a MergeOnlyPlugin
        # to merge the results automatically.
        if isinstance(targets, (list, tuple)) and len(targets) > 1:
            plugins = self._get_plugins(targets=targets, run_id=run_id)
            if len(set(plugins[d].data_kind_for(d) for d in targets)) == 1:
                temp_name = ('_temp_' + strax.deterministic_hash(targets))
                p = type(temp_name,
                         (strax.MergeOnlyPlugin,),
                         dict(depends_on=tuple(targets)))
                self.register(p)
                targets = (temp_name,)
            elif not allow_multiple:
                raise RuntimeError("Cannot automerge different data kinds!")
            elif (self.context_config['timeout'] > 7200 or (
                    self.context_config['allow_lazy'] and
                    not self.context_config['allow_multiprocess'])):
                # For allow_multiple we don't want allow this when in lazy mode
                # with long timeouts (lazy-mode is disabled if multiprocessing
                # so if that is activated, we can also continue)
                raise RuntimeError(f'Cannot allow_multiple in lazy mode or '
                                   f'with long timeouts.')

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
                timeout=self.context_config['timeout'],
                is_superrun=_is_superrun,).iter()

        try:
            _p, t_start, t_end = self._make_progress_bar(run_id,
                                                         targets=targets,
                                                         progress_bar=progress_bar)
            with _p as pbar:
                pbar.last_print_t = time.time()
                pbar.mbs = []
                for n_chunks, result in enumerate(strax.continuity_check(generator), 1):
                    seen_a_chunk = True
                    if not isinstance(result, strax.Chunk):
                        raise ValueError(f"Got type {type(result)} rather than "
                                         f"a strax Chunk from the processor!")
                    # Apply functions known to contexts if any.
                    result.data = self._apply_function(result.data,
                                                       run_id,
                                                       targets_list)

                    result.data = strax.apply_selection(
                        result.data,
                        selection_str=selection_str,
                        keep_columns=keep_columns,
                        drop_columns=drop_columns,
                        time_range=time_range,
                        time_selection=time_selection)
                    self._update_progress_bar(
                        pbar, t_start, t_end, n_chunks, result.end, result.nbytes)
                    yield result
            _p.close()

        except GeneratorExit:
            generator.throw(OutsideException(
                "Terminating due to an exception originating from outside "
                "strax's get_iter (which we cannot retrieve)."))

        except Exception as e:
            generator.throw(e)
            raise ValueError(f'Failed to process chunk {n_chunks}!')

        if not seen_a_chunk:
            if time_range is None:
                raise strax.DataCorrupted("No data returned!")
            raise ValueError(f"Invalid time range: {time_range}, "
                             "returned no chunks!")

    def _make_progress_bar(self, run_id, targets, progress_bar=True):
        """
        Make a progress bar for get_iter
        :param run_id, targets: run_id and targets
        :param progress_bar: Bool whether or not to display the progress bar
        :return: progress bar, t_start (run) and t_end (run)
        """
        try:
            t_start, t_end, = self.estimate_run_start_and_end(run_id, targets)
        except (AttributeError, KeyError, IndexError):
            # During testing some thing remain a secret
            t_start, t_end, = 0, float('inf')
        if t_end == float('inf'):
            progress_bar = False

        # Define nice progressbar format:
        bar_format = ("{desc}: "  # The loading plugin x
                      "|{bar}| "  # Bar that is being filled
                      "{percentage:.2f} % "  # Percentage
                      "[{elapsed}<{remaining}]"  # Time estimate
                      "{postfix}"  # Extra info
                      )
        description = f'Loading {"plugins" if targets[0].startswith("_temp") else targets}'
        pbar = tqdm(total=1,
                    desc=description,
                    bar_format=bar_format,
                    leave=True,
                    disable=not progress_bar)
        return pbar, t_start, t_end

    @staticmethod
    def _update_progress_bar(pbar, t_start, t_end, n_chunks, chunk_end, nbytes):
        """Do some tqdm voodoo to get the progress bar for st.get_iter"""
        if t_end - t_start > 0:
            fraction_done = (chunk_end - t_start) / (t_end - t_start)
            if fraction_done > .99:
                # Patch to 1 to not have a red pbar when very close to 100%
                fraction_done = 1
            pbar.n = np.clip(fraction_done, 0, 1)
        else:
            # Strange, start and endtime are the same, probably we don't
            # have data yet e.g. allow_incomplete == True.
            pbar.n = 0
        # Let's add the postfix which is the info behind the tqdm marker
        seconds_per_chunk = time.time() - pbar.last_print_t
        pbar.mbs.append((nbytes/1e6)/seconds_per_chunk)
        mbs = np.mean(pbar.mbs)
        if mbs < 1:
            rate = f'{mbs*1000:.1f} kB/s'
        else:
            rate = f'{mbs:.1f} MB/s'
        postfix = f'#{n_chunks} ({seconds_per_chunk:.2f} s). {rate}'
        pbar.set_postfix_str(postfix)
        pbar.update(0)

    def make(self, run_id: ty.Union[str, tuple, list],
             targets, save=tuple(),
             max_workers=None,
             _skip_if_built=True,
             **kwargs) -> None:
        """Compute target for run_id. Returns nothing (None).
        {get_docs}
        """
        kwargs.setdefault('progress_bar', False)

        # Multi-run support
        run_ids = strax.to_str_tuple(run_id)
        if len(run_ids) == 0:
            raise ValueError("Cannot build empty list of runs")
        if len(run_ids) > 1:
            return strax.multi_run(
                self.get_array, run_ids, targets=targets,
                throw_away_result=True, log=self.log,
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

        if kwargs.get('allow_multiple', False):
            raise RuntimeError('Cannot allow_multiple with get_array/get_df')

        if len(run_ids) > 1:
            results = strax.multi_run(
                self.get_array, run_ids, targets=targets,
                log=self.log,
                save=save, max_workers=max_workers, **kwargs)
        else:
            source = self.get_iter(
                run_ids[0],
                targets,
                save=save,
                max_workers=max_workers,
                **kwargs)
            results = [x.data for x in source]

        results = np.concatenate(results)
        return results

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

            NB: Additionally and independently, if there are any functions registered
            under context_config['apply_data_function'] these are applied first directly
            after loading the data.

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
        if kwargs.get('allow_multiple', False):
            raise RuntimeError('Cannot allow_multiple with accumulate')

        n_chunks = 0
        seen_data = False
        result = {'n_rows': 0}
        if fields is not None:
            fields = strax.to_str_tuple(fields)
        if function is None:
            def function(arr):
                return arr

            function_takes_fields = False

        for chunk in self.get_iter(run_id, targets,
                                   **kwargs):
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
                    if name not in fields:
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

    def get_zarr(self, run_ids, targets, storage='./strax_temp_data',
                 progress_bar=False, overwrite=True, **kwargs):
        """get persistent  arrays using zarr. This is useful when
            loading large amounts of data that cannot fit in memory
            zarr is very compatible with dask.
            Targets are loaded into separate arrays and runs are merged.
            the data is added to any existing data in the storage location.

        :param run_ids: (Iterable) Run ids you wish to load.
        :param targets: (Iterable) targets to load.
        :param storage: (str, optional) fsspec path to store array. Defaults to './strax_temp_data'.
        :param overwrite: (boolean, optional) whether to overwrite existing arrays for targets at given path.

        :returns zarr.Group: zarr group containing the persistant arrays available at
                        the storage location after loading the requested data
                        the runs loaded into a given array can be seen in the
                        array .attrs['RUNS'] field
        """
        import zarr
        context_hash = self._context_hash()
        kwargs_hash = strax.deterministic_hash(kwargs)
        root = zarr.open(storage, mode='w')
        group = root.require_group(context_hash+'/'+kwargs_hash, overwrite=overwrite)
        for target in strax.to_str_tuple(targets):
            idx = 0
            zarray = None
            if target in group:
                zarray = group[target]
                if not overwrite:
                    idx = zarray.size
            INSERTED = {}
            for run_id in strax.to_str_tuple(run_ids):
                if zarray is not None and run_id in zarray.attrs.get('RUNS', {}):
                    continue
                key = self.key_for(run_id, target)
                INSERTED[run_id] = dict(start_idx=idx, end_idx=idx, lineage_hash=key.lineage_hash)
                for chunk in self.get_iter(run_id, target, progress_bar=progress_bar, **kwargs):
                    end_idx = idx+chunk.data.size
                    if zarray is None:
                        dtype = [(d[0][1], )+d[1:] for d in chunk.dtype.descr]
                        zarray = group.create_dataset(target, shape=end_idx, dtype=dtype)
                    else:
                        zarray.resize(end_idx)
                    zarray[idx:end_idx] = chunk.data
                    idx = end_idx
                    INSERTED[run_id]['end_idx'] = end_idx
            zarray.attrs['RUNS'] = dict(zarray.attrs.get('RUNS', {}), **INSERTED)
        return group

    def key_for(self, run_id, target):
        """
        Get the DataKey for a given run and a given target plugin. The
        DataKey is inferred from the plugin lineage. The lineage can
        come either from the _fixed_plugin_cache or computed on the fly.

        :param run_id: run id to get
        :param target: data type to get
        :return: strax.DataKey of the target
        """
        if self._plugins_are_cached((target,)):
            context_hash = self._context_hash()
            if context_hash in self._fixed_plugin_cache:
                plugins = self._fixed_plugin_cache[self._context_hash()]
            else:
                # This once happened due to temp. plugins, should not happen again
                self.log.warning(f'Context hash changed to {context_hash} for '
                                 f'{self._plugin_class_registry}?')
                plugins = self._get_plugins((target,), run_id)
        else:
            plugins = self._get_plugins((target,), run_id)

        lineage = plugins[target].lineage
        return strax.DataKey(run_id, target, lineage)

    def get_meta(self, run_id, target) -> dict:
        """Return metadata for target for run_id, or raise DataNotAvailable
        if data is not yet available.

        :param run_id: run id to get
        :param target: data type to get
        """
        key = self.key_for(run_id, target)
        for sf in self._sorted_storage:
            try:
                return sf.get_metadata(key, **self._find_options)
            except strax.DataNotAvailable:
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
        for sf in self._sorted_storage:
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
        """
        Return whether data type target has been saved for run_id
        through any of the registered storage frontends.

        Note that even if False is returned, the data type may still be made
        with a trivial computation.
        """
        if isinstance(target, (tuple, list)):
            return all([self.is_stored(run_id, t, **kwargs)
                        for t in target])

        # If any new options given, replace the current context
        # with a temporary one
        if len(kwargs):
            # Comment below disables pycharm from inspecting the line below it
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        for sf in self._sorted_storage:
            if self._is_stored_in_sf(run_id, target, sf):
                return True
        # None of the frontends has the data
        return False

    def _check_forbidden(self):
        """Check that the forbid_creation_of config is of tuple type.
        Otherwise, try to make it a tuple"""
        self.context_config['forbid_creation_of'] = strax.to_str_tuple(
            self.context_config['forbid_creation_of'])

    def _apply_function(self,
                        chunk_data: np.ndarray,
                        run_id: ty.Union[str, tuple, list],
                        targets: ty.Union[str, tuple, list],
                        ) -> np.ndarray:
        """
        Apply functions stored in the context config to any data that is returned via
            get_array, get_df or accumulate. Functions stored in
            context_config['apply_data_function'] should take exactly two positional
            arguments: data and targets.
        :param data: Any type of data
        :param run_id: run_id of the data.
        :param targets: list/tuple of strings of data type names to get
        :return: the data after applying the function(s)
        """
        apply_functions = self.context_config['apply_data_function']
        if hasattr(apply_functions, '__call__'):
            # Apparently someone did not read the docstring and inserted
            # a single function instead of a list.
            apply_functions = [apply_functions]
        if not isinstance(apply_functions, (tuple, list)):
            raise ValueError(f"apply_data_function in context config should be tuple of "
                             f"functions. Instead got {apply_functions}")
        for function in apply_functions:
            if not hasattr(function, '__call__'):
                raise TypeError(f'apply_data_function in the context_config got '
                                f'{function} but expected callable function with two '
                                f'positional arguments: f(data, targets).')
            # Make sure that the function takes two arguments (data and targets)
            chunk_data = function(chunk_data, run_id, targets)
        return chunk_data

    def copy_to_frontend(self,
                         run_id: str,
                         target: str,
                         target_frontend_id: ty.Optional[int] = None,
                         target_compressor: ty.Optional[str] = None,
                         rechunk: bool = False):
        """
        Copy data from one frontend to another

        :param run_id: run_id
        :param target: target datakind
        :param target_frontend_id: index of the frontend that the data should go to
            in context.storage. If no index is specified, try all.
        :param target_compressor: if specified, recompress with this compressor.
        :param rechunk: allow re-chunking for saving
        """

        # NB! We don't want to use self._sorted_storage here since the order matters!
        if not self.is_stored(run_id, target):
            raise strax.DataNotAvailable(f'Cannot copy {run_id} {target} since it '
                                         f'does not exist')
        if len(strax.to_str_tuple(target)) > 1:
            raise ValueError(
                'copy_to_frontend only works for a single target at the time')
        if target_frontend_id is None:
            target_sf = self.storage
        elif len(self.storage) > target_frontend_id:
            # only write to selected other frontend
            target_sf = [self.storage[target_frontend_id]]
        else:
            raise ValueError(f'Cannot select {target_frontend_id}-th frontend as '
                             f'we only have {len(self.storage)} frontends!')

        # Figure out which of the frontends has the data. Raise error when none
        source_sf = self._get_source_sf(run_id, target, should_exist=True)

        # Keep frontends that:
        #  1. don't already have the data; and
        #  2. take the data; and
        #  3. are not readonly
        target_sf = [t_sf for t_sf in target_sf if
                     (not self._is_stored_in_sf(run_id, target, t_sf) and
                      t_sf._we_take(target) and
                      t_sf.readonly is False)]
        self.log.info(f'Copy data from {source_sf} to {target_sf}')
        if not len(target_sf):
            raise ValueError('No frontend to copy to! Perhaps you already stored '
                             'it or none of the frontends is willing to take it?')

        # Get the info from the source backend (s_be) that we need to fill
        # the target backend (t_be) with
        data_key = self.key_for(run_id, target)
        # This should never fail, we just tried
        s_be_str, s_be_key = source_sf.find(data_key)
        s_be = source_sf._get_backend(s_be_str)
        md = s_be.get_metadata(s_be_key)

        if target_compressor is not None:
            self.log.info(f'Changing compressor from {md["compressor"]} '
                           f'to {target_compressor}.')
            md.update({'compressor': target_compressor})

        for t_sf in target_sf:
            try:
                # Need to load a new loader each time since it's a generator
                # and will be exhausted otherwise.
                loader = s_be.loader(s_be_key)
                # Fill the target buffer
                t_be_str, t_be_key = t_sf.find(data_key, write=True)
                target_be = t_sf._get_backend(t_be_str)
                saver = target_be._saver(t_be_key, md)
                saver.save_from(loader, rechunk=rechunk)
            except NotImplementedError:
                # Target is not susceptible
                continue
            except strax.DataExistsError:
                raise strax.DataExistsError(
                    f'Trying to write {data_key} to {t_sf} which already exists, '
                    'do you have two storage frontends writing to the same place?')

    def get_source(self,
                   run_id: str,
                   target: str,
                   check_forbidden: bool = True,
                   ) -> ty.Union[set, None]:
        """
        For a given run_id and target get the stored bases where we can
        start processing from, if no base is available, return None.

        :param run_id: run_id
        :param target: target
        :param check_forbidden: Check that we are not requesting to make
            a plugin that is forbidden by the context to be created.
        :return: set of plugin names that are needed to start processing
            from and are needed in order to build this target.
        """
        try:
            return set(plugin_name
                       for plugin_name, plugin_stored in
                       self.stored_dependencies(run_id=run_id,
                                                target=target,
                                                check_forbidden=check_forbidden
                                                ).items()
                       if plugin_stored
                       )
        except strax.DataNotAvailable:
            return None

    def stored_dependencies(self,
                            run_id: str,
                            target: ty.Union[str, list, tuple],
                            check_forbidden: bool = True,
                            _targets_stored: ty.Optional[dict] = None,
                            ) -> ty.Optional[dict]:
        """
        For a given run_id and target(s) get a dictionary of all the datatypes that:

        :param run_id: run_id
        :param target: target or a list of targets
        :param check_forbidden: Check that we are not requesting to make
            a plugin that is forbidden by the context to be created.
        :return: dictionary of data types (keys) required for building
            the requested target(s) and if they are stored (values)
        :raises strax.DataNotAvailable: if there is at least one data
            type that is not stored and has no dependency or if it
            cannot be created
        """
        if _targets_stored is None:
            _targets_stored = dict()

        targets = strax.to_str_tuple(target)
        if len(targets) > 1:
            # Multiple targets, do them all
            for dep in targets:
                self.stored_dependencies(run_id,
                                         dep,
                                         check_forbidden=check_forbidden,
                                         _targets_stored=_targets_stored,
                                         )
            return _targets_stored

        # Make sure we have the string not ('target',)
        target = targets[0]

        if target in _targets_stored:
            return

        this_target_is_stored = self.is_stored(run_id, target)
        _targets_stored[target] = this_target_is_stored

        if this_target_is_stored:
            return _targets_stored

        # Need to init the class e.g. if we want to allow depends on like this:
        # https://github.com/XENONnT/cutax/blob/d7ec0685650d03771fef66507fd6882676151b9b/cutax/cutlist.py#L33  # noqa
        plugin = self._plugin_class_registry[target]()
        dependencies = strax.to_str_tuple(plugin.depends_on)
        if not dependencies:
            raise strax.DataNotAvailable(f'Lowest level dependency {target} is not stored')

        forbidden = strax.to_str_tuple(self.context_config['forbid_creation_of'])
        forbidden_warning = (
            'For {run_id}:{target}, you are not allowed to make {dep} and '
            'it is not stored. Disable check with check_forbidden=False'
        )
        if check_forbidden and target in forbidden:
            raise strax.DataNotAvailable(
                forbidden_warning.format(run_id=run_id, target=target, dep=target,))

        self.stored_dependencies(run_id,
                                 target=dependencies,
                                 check_forbidden=check_forbidden,
                                 _targets_stored=_targets_stored,
                                 )
        return _targets_stored

    def _is_stored_in_sf(self, run_id, target,
                         storage_frontend: strax.StorageFrontend) -> bool:
        """
        :param run_id, target: run_id, target
        :param storage_frontend: strax.StorageFrontend to check if it has the
        requested datakey for the run_id and target.
        :return: if the frontend has the key or not.
        """
        key = self.key_for(run_id, target)
        try:
            storage_frontend.find(key, **self._find_options)
            return True
        except strax.DataNotAvailable:
            return False

    def _get_source_sf(self, run_id, target, should_exist=False):
        """
        Get the source storage frontend for a given run_id and target
        :param run_id, target: run_id, target
        :param should_exist: Raise a ValueError if we cannot find one
        (e.g. we already checked the data is stored)
        :return: strax.StorageFrontend or None (when raise_error is
        False)
        """
        for sf in self._sorted_storage:
            if self._is_stored_in_sf(run_id, target, sf):
                return sf
        if should_exist:
            raise ValueError('This cannot happen, we just checked that this '
                             'run should be stored?!?')

    def get_save_when(self, target: str) -> ty.Union[strax.SaveWhen, int]:
        """for a given plugin, get the save when attribute either being a
        dict or a number"""
        plugin_class = self._plugin_class_registry[target]
        save_when = plugin_class.save_when
        if isinstance(save_when, immutabledict):
            save_when = save_when[target]
        if not isinstance(save_when, (IntEnum, int)):
            raise ValueError(f'SaveWhen of {plugin_class} should be IntEnum '
                             f'or immutabledict')
        return save_when

    def provided_dtypes(self, runid='0'):
        """
        Summarize dtype information provided by this context
        :return: dictionary of provided dtypes with their corresponding lineage hash, save_when, version
        """
        hashes = set([(data_type,
                       self.key_for(runid, data_type).lineage_hash,
                       self.get_save_when(data_type),
                       plugin.__version__)
                      for plugin in self._plugin_class_registry.values()
                      for data_type in plugin.provides])

        return {data_type: dict(hash=_hash,
                                save_when=save_when.name,
                                version=version)
                for data_type, _hash, save_when, version in hashes}

    @classmethod
    def add_method(cls, f):
        """Add f as a new Context method"""
        setattr(cls, f.__name__, f)


select_docs = """
:param selection_str: Query string or sequence of strings to apply.
:param keep_columns: Array field/dataframe column names to keep. 
    Useful to reduce amount of data in memory. (You can only specify 
    either keep or drop column.)
:param drop_columns: Array field/dataframe column names to drop. 
    Useful to reduce amount of data in memory. (You can only specify 
    either keep or drop column.)
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
:param multi_run_progress_bar: Display a progress bar for loading multiple runs
"""

get_docs = """
:param run_id: run id to get
:param targets: list/tuple of strings of data type names to get
:param save: extra data types you would like to save
    to cache, if they occur in intermediate computations.
    Many plugins save automatically anyway.
:param max_workers: Number of worker threads/processes to spawn.
    In practice more CPUs may be used due to strax's multithreading.
:param allow_multiple: Allow multiple targets to be computed
    simultaneously without merging the results of the target. This can
    be used when mass producing plugins that are not of the same
    datakind. Don't try to use this in get_array or get_df because the
    data is not returned.
:param add_run_id_field: Boolean whether to add a run_id field in case
    of multi-runs.
:param run_id_as_bytes: Boolean if true uses byte string instead of an
    unicode string added to a multi-run array. This can save a lot of 
    memory when loading many runs.
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
