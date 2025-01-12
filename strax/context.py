import time
import logging
import fnmatch
import itertools
import json
from functools import partial
from copy import deepcopy
import types
import typing as ty
from enum import IntEnum
import datetime
import inspect
from collections import defaultdict

from immutabledict import immutabledict
import numpy as np
import pandas as pd
import strax
from strax import CutList


export, __all__ = strax.exporter()
__all__.extend(["RUN_DEFAULTS_KEY"])

RUN_DEFAULTS_KEY = "strax_defaults"
TEMP_DATA_TYPE_PREFIX = "_temp_"

# use tqdm as loaded in utils (from tqdm.notebook when in a jupyter env)
tqdm = strax.utils.tqdm


@strax.takes_config(
    strax.Option(
        name="fuzzy_for",
        default=tuple(),
        type=tuple,
        help=(
            "Tuple or string of plugin names for which no checks for version, "
            "providing plugin, and config will be performed when "
            "looking for data."
        ),
    ),
    strax.Option(
        name="fuzzy_for_options",
        default=tuple(),
        type=tuple,
        help="Tuple of config options for which no checks will be performed when looking for data.",
    ),
    strax.Option(
        name="allow_incomplete",
        default=False,
        type=bool,
        help="Allow loading of incompletely written data, if the storage systems support it",
    ),
    strax.Option(
        name="allow_rechunk",
        default=True,
        type=bool,
        help="Allow rechunking of data during writing.",
    ),
    strax.Option(
        name="allow_multiprocess",
        default=False,
        type=bool,
        help="Allow multiprocessing.If False, will use multithreading only.",
    ),
    strax.Option(
        name="allow_shm",
        default=False,
        type=bool,
        help="Allow use of /dev/shm for interprocess communication.",
    ),
    strax.Option(
        name="allow_lazy",
        default=True,
        type=bool,
        help=(
            'Allow "lazy" processing. Saves memory, but incompatible '
            "with multiprocessing and perhaps slightly slower."
        ),
    ),
    strax.Option(
        name="forbid_creation_of",
        default=tuple(),
        type=tuple,
        help=(
            "If any of the following datatypes is requested to be "
            "created, throw an error instead. Useful to limit "
            "descending too far into the dependency graph."
        ),
    ),
    strax.Option(
        name="store_run_fields",
        default=tuple(),
        type=tuple,
        help="Tuple of run document fields to store during scan_run.",
    ),
    strax.Option(
        name="check_available",
        default=tuple(),
        type=tuple,
        help="Tuple of data types to scan availability for during scan_run.",
    ),
    strax.Option(
        name="max_messages",
        default=4,
        type=int,
        help=(
            "Maximum number of mailbox messages, i.e. size of buffer "
            "between plugins. Too high = RAM blows up. "
            "Too low = likely deadlocks."
        ),
    ),
    strax.Option(
        name="timeout",
        default=24 * 3600,
        type=int,
        help=(
            "Terminate processing if any one mailbox receives "
            "no result for more than this many seconds"
        ),
    ),
    strax.Option(
        name="saver_timeout",
        default=900,
        type=int,
        help=(
            "Max time [s] a saver can take to store a result. Set "
            "high for slow compression algorithms."
        ),
    ),
    strax.Option(
        name="use_per_run_defaults",
        default=False,
        type=bool,
        help=(
            "Scan the run db for per-run defaults. "
            "This is an experimental strax feature that will "
            "possibly be removed, see issue #246"
        ),
    ),
    strax.Option(
        name="free_options",
        default=tuple(),
        type=(tuple, list),
        help=(
            "Do not warn if any of these options are passed, "
            "even when no registered plugin takes them."
        ),
    ),
    strax.Option(
        name="apply_data_function",
        default=tuple(),
        type=(tuple, list, ty.Callable),
        help=(
            "Apply a function to the data prior to returning the"
            "data. The function should take three positional arguments: "
            "func(<data>, <run_id>, <targets>)."
        ),
    ),
    strax.Option(
        name="write_superruns",
        default=False,
        type=bool,
        help='If True, save superruns as rechunked "new" data.',
    ),
)
@export
class Context:
    """Context for strax analysis.

    A context holds info on HOW to process data, such as which plugins provide what data types,
    where to store which results, and configuration options for the plugins.

    You start all strax processing through a context.

    """

    config: dict
    context_config: dict

    runs: ty.Optional[pd.DataFrame] = None
    _fixed_plugin_cache: ty.Optional[dict] = None
    _fixed_level_cache: ty.Optional[dict] = None
    _run_defaults_cache: dict
    storage: ty.List[strax.StorageFrontend]

    processors: ty.Mapping[str, strax.BaseProcessor]

    def __init__(
        self, storage=None, config=None, register=None, register_all=None, processors=None, **kwargs
    ):
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
        :param processors: A mapping of processor names to classes to use for
            data processing.
        Any additional kwargs are considered Context-specific options; see
        Context.takes_config.

        """
        self.log = logging.getLogger("strax")

        if storage is None:
            storage = ["./strax_data"]
        if not isinstance(storage, (list, tuple)):
            storage = [storage]
        self.storage = [strax.DataDirectory(s) if isinstance(s, str) else s for s in storage]

        self._plugin_class_registry = dict()
        self._run_defaults_cache = dict()

        self.set_config(config, mode="replace")
        self.set_context_config(kwargs, mode="replace")

        if register_all is not None:
            self.register_all(register_all)
        if register is not None:
            self.register(register)

        if processors is None:
            processors = strax.PROCESSORS

        if isinstance(processors, str):
            processors = [processors]

        if isinstance(processors, (list, tuple)):
            ps = {}
            for processor in processors:
                if isinstance(processor, str) and processor in strax.PROCESSORS:
                    ps[processor] = strax.PROCESSORS[processor]
                elif isinstance(processor, strax.BaseProcessor):
                    ps[processor.__name__] = processor
                else:
                    raise ValueError(f"Unknown processor {processor}")
            processors = ps

        self.processors = processors

    def new_context(
        self,
        storage=tuple(),
        config=None,
        register=None,
        register_all=None,
        processors=None,
        replace=False,
        **kwargs,
    ):
        """Return a new context with new setting adding to those in this context.

        :param replace: If True, replaces settings rather than adding them. See Context.__init__ for
            documentation on other parameters.

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
            config = strax.combine_configs(self.config, config, mode="update")
            kwargs = strax.combine_configs(self.context_config, kwargs, mode="update")

        new_c = Context(storage=storage, config=config, processors=processors, **kwargs)
        if not replace:
            new_c._plugin_class_registry = self._plugin_class_registry.copy()
        new_c.register_all(register_all)
        new_c.register(register)
        return new_c

    def set_config(self, config=None, mode="update"):
        """Set new configuration options.

        :param config: dict of new options
        :param mode: can be either
            - update: Add to or override current options in context
            - setdefault: Add to current options, but do not override
            - replace: Erase config, then set only these options

        """
        if not hasattr(self, "config"):
            self.config = dict()
        self.config = strax.combine_configs(old_config=self.config, new_config=config, mode=mode)

    def set_context_config(self, context_config=None, mode="update"):
        """Set new context configuration options.

        :param context_config: dict of new context configuration options
        :param mode: can be either
            - update: Add to or override current options in context
            - setdefault: Add to current options, but do not override
            - replace: Erase config, then set only these options

        """
        if not hasattr(self, "context_config"):
            self.context_config = dict()

        new_config = strax.combine_configs(
            old_config=self.context_config, new_config=context_config, mode=mode
        )

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

        :param plugin_class: class inheriting from strax.Plugin. You can also pass a sequence of
            plugins to register, but then you must omit the provides argument. If a plugin class
            omits the .provides attribute, we will construct one from its class name (CamelCase ->
            snake_case) Returns plugin_class (so this can be used as a decorator)

        """
        if isinstance(plugin_class, (tuple, list)):
            # Shortcut for multiple registration
            for x in plugin_class:
                self.register(x)
            return

        if not issubclass(plugin_class, strax.Plugin):
            raise ValueError(f"Can only register subclasses of strax.Plugin, not {plugin_class}!")

        if not hasattr(plugin_class, "provides"):
            # No output name specified: construct one from the class name
            snake_name = strax.camel_to_snake(plugin_class.__name__)
            plugin_class.provides = (snake_name,)

        # Ensure plugin_class.provides is a tuple
        if isinstance(plugin_class.provides, str):
            plugin_class.provides = tuple([plugin_class.provides])

        # Register the plugin for all datatypes it provides,
        # tracking which plugins we booted out.
        deregistered = []
        for p in plugin_class.provides:
            old_plugin_class = self._plugin_class_registry.get(p, None)
            if old_plugin_class and old_plugin_class != plugin_class:
                deregistered.append(old_plugin_class)
            self._plugin_class_registry[p] = plugin_class

        # If we booted a plugin from a datatype, we must boot it from other
        # datatypes it makes too, to preserve a one-to-one mapping between
        # datatypes and registered plugins.
        for old_plugin in set(deregistered):
            for d in old_plugin.provides:
                currently_registered = self._plugin_class_registry.get(d)
                if old_plugin == currently_registered:
                    # Must be equal here, because we are only looking for the remanants which were
                    # not overwritten above.
                    self.log.warning(
                        "Provides of multi-output plugins overlap, deregister old plugins"
                        f" {old_plugin}."
                    )
                    del self._plugin_class_registry[d]

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
                        default = items.get_default("0")  # Have to pass will be changed.
                        new_default = new_items.get_default("0")
                        if default == new_default:
                            continue
                        else:
                            mes = (
                                "Two plugins have a different default value"
                                " for the same option. The option"
                                f' "{new_option}" in "{plugin.__name__}" takes'
                                f' as a default "{default}" while in'
                                f' "{plugin_class.__name__}" the default value'
                                f' is set to "{new_default}". Please change'
                                " one of the defaults."
                            )
                            raise ValueError(mes)

                except strax.InvalidConfiguration:
                    # These are option which are inherited from context options.
                    pass

        return plugin_class

    def purge_unused_configs(self):
        """Purge unused configs from the context."""
        all_opts = set().union(
            *[pc.takes_config.keys() for pc in self._plugin_class_registry.values()]
        )
        waiting_for = []
        for k in self.config:
            if not (k in all_opts or k in self.context_config["free_options"]):
                self.log.warning(f"Option {k} purged from context config as it is not used.")
                waiting_for.append(k)
        for k in waiting_for:
            del self.config[k]

    def deregister_plugins_with_missing_dependencies(self):
        """Deregister plugins in case a data_type the plugin depends on is not provided by any other
        plugin."""
        registry_changed = True
        while registry_changed:
            all_provides = set()
            plugins_to_deregister = []

            for p in self._plugin_class_registry.values():
                all_provides |= set(p.provides)

            for p_key, p in self._plugin_class_registry.items():
                requires = set(strax.to_str_tuple(p().depends_on))
                if not requires.issubset(all_provides):
                    plugins_to_deregister.append(p_key)

            for p_key in plugins_to_deregister:
                self.log.info(f"Deregister {p_key}")
                del self._plugin_class_registry[p_key]

            if not len(plugins_to_deregister):
                registry_changed = False

    def get_data_kinds(self) -> ty.Tuple:
        """Return two dictionaries:
        1. one with all available data_kind as key and their data_types(list) as values
        2. one with all available data_type as key and their data_kind(str) as values
        """
        data_kind_collection: ty.Dict[str, ty.List] = dict()
        data_type_collection: ty.Dict[str, str] = dict()
        for data_type in self._plugin_class_registry.keys():
            plugin = self.__get_plugin("0", data_type)
            if isinstance(plugin.data_kind, (dict, immutabledict)):
                data_kind = plugin.data_kind[data_type]
            else:
                data_kind = plugin.data_kind
            data_kind_collection.setdefault(data_kind, [])
            data_kind_collection[data_kind].append(data_type)
            data_type_collection[data_type] = data_kind
        return data_kind_collection, data_type_collection

    def search_field(
        self,
        pattern: str,
        include_code_usage: bool = True,
        return_matches: bool = False,
    ):
        """Find and print which plugin(s) provides a field that matches pattern (fnmatch).

        :param pattern: pattern to match, e.g. 'time' or 'tim*'
        :param include_code_usage: Also include the code occurrences of the fields that match the
            pattern.
        :param return_matches: If set, return a dictionary with the matching fields and the
            occurrences in code.
        :return: when return_matches is set, return a dictionary with the matching fields and the
            occurrences in code. Otherwise, we are not returning anything and just print the results

        """
        cache = dict()
        field_matches = defaultdict(list)
        code_matches = dict()
        for data_type in sorted(list(self._plugin_class_registry.keys())):
            if data_type not in cache:
                cache.update(self._get_plugins((data_type,), run_id="0"))
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

    def search_field_usage(
        self, search_string: str, plugin: ty.Union[strax.Plugin, ty.List[strax.Plugin], None] = None
    ) -> ty.List[str]:
        """Find and return which plugin(s) use a given field.

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
                    for line in inspect.getsource(class_attribute).split("\n"):
                        if search_string in line:
                            if plug.__class__.__name__ == "type":
                                # Make sure we have the instance, not the class:
                                # >>> class A: pass
                                # >>> A.__class__.__name__
                                # 'type'
                                # >>> A().__class__.__name__
                                # 'A'
                                plug = plug()  # type: ignore
                            result += [f"{plug.__class__.__name__}.{attribute_name}"]
                            # Likely to be used several other times
                            break
        return result

    def show_config(self, data_type=None, pattern="*", run_id="9" * 20):
        """Return configuration options that affect data_type.

        :param data_type: Data type name
        :param pattern: Show only options that match (fnmatch) pattern
        :param run_id: run id to use for run-dependent config options. If omitted, will show
            defaults active for new runs.

        """
        r = []
        if data_type is None:
            # search for context options
            it = [["Context", self]]
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
                r.append(
                    dict(
                        option=opt.name,
                        default=default,
                        current=c.get(opt.name, strax.OMITTED),
                        applies_to=(p.provides if d != "Context" else d),
                        help=opt.help,
                    )
                )
        if len(r):
            return pd.DataFrame(r, columns=r[0].keys())
        return pd.DataFrame([])

    def lineage(self, run_id, data_type, chunk_number=None):
        """Return lineage dictionary for data_type and run_id, based on the options in this
        context."""
        return self._get_plugins((data_type,), run_id, chunk_number=chunk_number)[data_type].lineage

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

    def register_cut_list(self, cut_list):
        """Register cut lists to strax context.

        :param cut_list: cut lists to be registered. can be cutlist object or list/tuple of cutlist
            objects

        """
        assert not isinstance(
            cut_list, str
        ), "Please don't put string... use cutlist object or list/tuple of cutlist objects"
        if hasattr(cut_list, "__len__"):
            for _cut_list in cut_list:
                self.register_cut_list(_cut_list)
        else:
            for cut in cut_list.cuts:
                # maybe cutlist within cutlist?
                if CutList in cut.__bases__:
                    self.register_cut_list(cut)
                else:
                    self.register(cut)
            self.register(cut_list)

    def data_itemsize(self, data_type: str) -> int:
        """Return size of a single item of data_type in bytes."""
        p = self._get_plugins((data_type,), run_id="0")[data_type]
        return p.dtype_for(data_type).itemsize

    def data_info(self, data_type: str) -> pd.DataFrame:
        """Return pandas DataFrame describing fields in data_type."""
        p = self._get_plugins((data_type,), run_id="0")[data_type]
        display_headers = ["Field name", "Data type", "Comment"]
        result = []
        for name, dtype in strax.utils.unpack_dtype(p.dtype_for(data_type)):
            if isinstance(name, tuple):
                title, name = name
            else:
                title = ""
            result.append([name, dtype, title])
        return pd.DataFrame(result, columns=display_headers)

    def get_single_plugin(self, run_id, data_type, chunk_number=None):
        """Return a single fully initialized plugin that produces data_type for run_id.

        For use in custom processing.

        """
        plugin = self._get_plugins((data_type,), run_id, chunk_number=chunk_number)[data_type]
        self._set_plugin_config(plugin, run_id, tolerant=False)
        plugin.setup()
        return plugin

    @staticmethod
    def _process_superrun_id(run_id):
        if not isinstance(run_id, str):
            raise ValueError(f"run_id {run_id} must be str, but got {type(run_id)}")
        if run_id.startswith("_"):
            _run_id = run_id[1:]
        else:
            _run_id = run_id
        return _run_id

    def _set_plugin_config(self, p, run_id, tolerant=True):
        # Explicit type check, since if someone calls this with
        # plugin CLASSES, funny business might ensue
        _run_id = self._process_superrun_id(run_id)
        assert isinstance(p, strax.Plugin)
        config = self.config.copy()
        for opt in p.takes_config.values():
            try:
                opt.validate(config, run_id=_run_id, run_defaults=self.run_defaults(_run_id))
            except strax.InvalidConfiguration:
                if not tolerant:
                    raise

        p.config = {k: v for k, v in config.items() if k in p.takes_config}

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

                    mes = (
                        f'Cannot find "{parent_name}" among the options of the parent.'
                        f" Either you specified by accident {option_name} as child option"
                        " or you specified the wrong parent_option_name. Have you specified "
                        "the correct parent option name?"
                    )
                    assert parent_name in p.config, mes
                    p.config[parent_name] = option_value

    def _context_hash(self):
        """Dump the current config + plugin class registry to a hash as a sanity check for building
        the _fixed_plugin_cache.

        If any item changes in the config, so does this hash.

        """
        # do not set _base_hash_on_config as an attribute of Context
        # because it might be changed across threads
        _base_hash_on_config = deepcopy(self.config)
        # Also take into account the versions of the plugins registered
        _base_hash_on_config.update(
            {
                data_type: (plugin.version(), plugin.compressor, plugin.input_timeout)
                for data_type, plugin in self._plugin_class_registry.items()
                if not data_type.startswith(TEMP_DATA_TYPE_PREFIX)
            }
        )
        return strax.deterministic_hash(_base_hash_on_config)

    def _plugins_are_cached(
        self,
        targets: ty.Union[ty.Tuple[str], ty.List[str]],
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ) -> bool:
        """Check if all the requested targets are in the _fixed_plugin_cache."""
        if (
            self.context_config["use_per_run_defaults"]
            or self._fixed_plugin_cache is None
            or chunk_number is not None
        ):
            # There is no point in caching if plugins (lineage) can
            # change per run or the cache is empty.
            return False

        context_hash = self._context_hash()
        if context_hash not in self._fixed_plugin_cache:
            return False
        plugin_cache = self._fixed_plugin_cache[context_hash]
        return all([t in plugin_cache for t in targets])

    def _plugins_to_cache(
        self,
        plugins: dict,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ) -> None:
        if self.context_config["use_per_run_defaults"] or chunk_number is not None:
            # There is no point in caching if plugins (lineage) can change per run
            return
        context_hash = self._context_hash()
        if self._fixed_plugin_cache is None:
            self._fixed_plugin_cache = {context_hash: dict()}
        elif context_hash not in self._fixed_plugin_cache:
            # Create a new cache every time the hash is not matching to
            # save memory. If a config changes, building the cache again
            # should be fast, we just need to track which cache to use.
            self.log.info("Replacing context._fixed_plugin_cache since plugins/versions changed")
            self._fixed_plugin_cache = {context_hash: dict()}
        for target, plugin in plugins.items():
            self._fixed_plugin_cache[context_hash][target] = plugin

    def __get_requested_plugins_from_cache(
        self,
        run_id: str,
        targets: ty.Tuple[str],
    ) -> ty.Dict[str, strax.Plugin]:
        """Load requested plugins from the plugin_cache.

        Doubly underscored since we don't do any key-checks etc here. Please be very careful of
        using it since no check is done.

        """
        requested_plugins = {}
        cached_plugins = self._fixed_plugin_cache[self._context_hash()]  # type: ignore
        for target, plugin in cached_plugins.items():
            if target in requested_plugins:
                # If e.g. target is already seen because the plugin is
                # multi output
                continue

            requested_p = plugin.__copy__()
            requested_p.run_id = run_id

            # Re-use only one instance if the plugin is multi output
            for provides in strax.to_str_tuple(requested_p.provides):
                requested_plugins[provides] = requested_p

        # Finally, fix the dtype.
        for plugin in requested_plugins.values():
            plugin.fix_dtype()

        requested_plugins = {i: v for i, v in requested_plugins.items() if i in targets}
        return requested_plugins

    def _get_plugins(
        self,
        targets: ty.Union[ty.Tuple[str], ty.List[str]],
        run_id: str,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ) -> ty.Dict[str, strax.Plugin]:
        """Return dictionary of plugin instances necessary to compute targets from scratch.

        For a plugin that produces multiple outputs, we make only a single instance, which is
        referenced under multiple keys in the output dict.

        """
        # Check all config options are taken by some registered plugin class
        # (helps spot typos)
        all_opts = set().union(
            *[pc.takes_config.keys() for pc in self._plugin_class_registry.values()]
        )
        for k in self.config:
            if not (k in all_opts or k in self.context_config["free_options"]):
                self.log.warning(f"Option {k} not taken by any registered plugin")

        plugins = {}
        targets = list(targets)
        safety_counter = 0
        while targets and safety_counter < 10_000:
            safety_counter += 1
            targets = list(set(targets))  # Remove duplicates from list.
            target = targets.pop(0)
            if target in plugins:
                continue

            target_plugin = self.__get_plugin(run_id, target, chunk_number=chunk_number)
            for provides in target_plugin.provides:
                plugins[provides] = target_plugin
            targets += list(target_plugin.depends_on)

        _not_all_plugins_initalized = (safety_counter == 10_000) & len(targets)
        if _not_all_plugins_initalized:
            raise ValueError(
                "Could not initalize all plugins to compute target from scratch. "
                f"The reamining targets missing are: {targets}"
            )

        return plugins

    def __get_plugin(
        self,
        run_id: str,
        data_type: str,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ):
        """Get single plugin either from cache or initialize it."""
        # Check if plugin for data_type is already cached
        if self._plugins_are_cached((data_type,), chunk_number=chunk_number):
            cached_plugins = self.__get_requested_plugins_from_cache(run_id, (data_type,))
            target_plugin = cached_plugins[data_type]
            return target_plugin

        if data_type not in self._plugin_class_registry:
            raise KeyError(f"No plugin class registered that provides {data_type}")

        plugin = self._plugin_class_registry[data_type]()

        plugin.run_id = run_id

        # The plugin may not get all the required options here
        # but we don't know if we need the plugin yet
        self._set_plugin_config(plugin, run_id, tolerant=True)

        plugin.deps = {
            d_depends: self.__get_plugin(run_id, d_depends, chunk_number=chunk_number)
            for d_depends in plugin.depends_on
        }

        self.__add_lineage_to_plugin(run_id, plugin, chunk_number=chunk_number)

        if not hasattr(plugin, "data_kind") and not plugin.multi_output:
            if len(plugin.depends_on):
                # Assume data kind is the same as the first dependency
                first_dep = plugin.depends_on[0]
                plugin.data_kind = plugin.deps[first_dep].data_kind_for(first_dep)
            else:
                # No dependencies: assume provided data kind and
                # data type are synonymous
                plugin.data_kind = plugin.provides[0]

        plugin.fix_dtype()

        # Add plugin to cache
        self._plugins_to_cache(
            {data_type: plugin for data_type in plugin.provides}, chunk_number=chunk_number
        )

        return plugin

    @staticmethod
    def _check_chunk_number(chunk_number: ty.List[int]):
        """Check if the chunk_number is a list of consecutive integers."""
        mask = isinstance(chunk_number, list)
        mask &= all([isinstance(x, int) for x in chunk_number])
        if not mask:
            raise ValueError(f"chunk_number should be a list of integers, but got {chunk_number}")

        # Check if the difference between adjacent elements is exactly one
        for i in range(len(chunk_number) - 1):
            if chunk_number[i + 1] - chunk_number[i] != 1:
                raise ValueError(
                    "chunk_number should be a list of consecutive integers, "
                    f"but got {chunk_number}"
                )

    def __add_lineage_to_plugin(
        self,
        run_id,
        plugin,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ):
        """Adds lineage to plugin in place.

        Also adds parent infromation in case of a child plugin.

        """
        last_provide = [d_provides for d_provides in plugin.provides][-1]

        if plugin.child_plugin:
            # Plugin is a child of another plugin, hence we have to
            # drop the parents config from the lineage
            configs = {}

            # Get all parent options which are overwritten by a child:
            parent_options = [
                option.parent_option_name
                for option in plugin.takes_config.values()
                if option.child_option
            ]

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
            for parent_class in plugin.__class__.__bases__:
                configs[parent_class.__name__] = parent_class.version()

        else:
            configs = {
                option: setting
                for option, setting in plugin.config.items()
                if plugin.takes_config[option].track
            }

        # Set chunk_number in the lineage
        if chunk_number is not None:
            for d_depends in plugin.depends_on:
                if d_depends in chunk_number:
                    not_allowed_plugins = (strax.LoopPlugin, strax.OverlapWindowPlugin)
                    if issubclass(plugin.__class__, not_allowed_plugins):
                        raise ValueError(
                            f"Can not assign chunk_number for {plugin.__class__} "
                            f"because it is subclass of one of {not_allowed_plugins}!"
                        )
                    configs.setdefault("chunk_number", {})
                    if d_depends in configs["chunk_number"]:
                        raise ValueError(
                            f"Chunk number for {d_depends} is already set in the lineage"
                        )
                    self._check_chunk_number(chunk_number[d_depends])
                    configs["chunk_number"][d_depends] = chunk_number[d_depends]

        plugin.lineage = {last_provide: (plugin.__class__.__name__, plugin.version(), configs)}

        # This is why the lineage of a plugin contains all its dependencies
        for d_depends in plugin.depends_on:
            plugin.lineage.update(plugin.deps[d_depends].lineage)

    def _per_run_default_allowed_check(self, option_name, option):
        """Check if an option of a registered plugin is allowed."""
        per_run_default = option.default_by_run != strax.OMITTED
        not_overwritten = option_name not in self.config
        per_run_is_forbidden = not self.context_config["use_per_run_defaults"]
        if per_run_default and not_overwritten and per_run_is_forbidden:
            raise strax.InvalidConfiguration(
                f"{option_name} is specified as a per-run-default which is not "
                "allowed by the context"
            )

    @staticmethod
    def _get_end_targets(plugins: dict) -> ty.Tuple[str]:
        """Get the datatype that is provided by a plugin but not depended on by any other plugin."""
        provides = [prov for p in plugins.values() for prov in strax.to_str_tuple(p.provides)]
        depends_on = [dep for p in plugins.values() for dep in strax.to_str_tuple(p.depends_on)]
        uniques = list(set(provides) - set(depends_on))
        return strax.to_str_tuple(uniques)

    @property
    def _find_options(self):
        # The plugin settings in the lineage are stored with the last
        # plugin provides name as a key. This can be quite confusing
        # since e.g. to be fuzzy for the peaklets settings the user has
        # to specify fuzzy_for=('lone_hits'). Here a small work around
        # to change this and not to reprocess the entire data set.
        fuzzy_for_keys = strax.to_str_tuple(self.context_config["fuzzy_for"])
        last_provides = []
        for key in fuzzy_for_keys:
            last_provides.append(self._plugin_class_registry[key].provides[-1])
        last_provides = tuple(last_provides)

        return dict(
            fuzzy_for=last_provides,
            fuzzy_for_options=self.context_config["fuzzy_for_options"],
            allow_incomplete=self.context_config["allow_incomplete"],
        )

    @property
    def _sorted_storage(self) -> ty.List[strax.StorageFrontend]:
        """Simple ordering of the storage frontends on the fly when e.g. looking for data.

        This allows us to use the simple self.storage as a simple list without asking users to keep
        any particular order in mind. Return the fastest first and try loading from it

        """
        return sorted(self.storage, key=lambda x: x.storage_type)

    def writable_storage(self) -> ty.List[strax.StorageFrontend]:
        """Return list of writable storage frontends."""
        return [s for s in self.storage if not s.readonly]

    def _get_partial_loader_for(self, key, time_range=None, chunk_number=None):
        """Get partial loaders to allow loading data later.

        :param key: strax.DataKey
        :param time_range: 2-length arraylike of (start, exclusive end) of row numbers to get.
            Default is None, which means get the entire run.
        :param chunk_number: number of the chunk for data specified by strax.DataKey. This chunck is
            loaded exclusively.
        :return: partial object

        """
        for sf in self._sorted_storage:
            try:
                # Partial is clunky... but allows specifying executor later
                # Since it doesn't run until later, we must do a find now
                # that we can still handle DataNotAvailable
                sf.find(key, **self._find_options)
                return partial(
                    sf.loader,
                    key,
                    time_range=time_range,
                    chunk_number=chunk_number,
                    **self._find_options,
                )
            except strax.DataNotAvailable:
                continue
        return False

    def get_components(
        self,
        run_id: str,
        targets=tuple(),
        save=tuple(),
        time_range=None,
        chunk_number=None,
        multi_run_progress_bar=False,
        combining=False,
    ) -> strax.ProcessorComponents:
        """Return components for setting up a processor.

        {get_docs}

        """
        save = strax.to_str_tuple(save)
        targets = strax.to_str_tuple(targets)

        for t in targets:
            if len(t) == 1:
                raise ValueError(f"Plugin names must be more than one letter, not {t}")

        is_superrun = run_id.startswith("_")
        if len(targets) > 1 and combining:
            raise ValueError("Combining subruns is only supported for a single target")
        if is_superrun and chunk_number is not None:
            raise ValueError("Per chunk processing is only allowed when not processing superrun.")
        if not is_superrun and combining:
            raise ValueError("Combining subruns is only supported for superruns.")

        sources = set().union(
            *[s for s in (self.get_source(run_id, target) for target in targets) if s is not None]
        )
        if chunk_number is not None:
            chunk_number_keys = set(chunk_number.keys())
            if not chunk_number_keys <= sources:
                self.log.warning(
                    f"Chunk number is specified for dependencies {chunk_number_keys} "
                    f"but {targets} are made from stored dependencies {sources}. "
                    "So some values in chunk_number will be ignored."
                )

        plugins = self._get_plugins(targets, run_id, chunk_number=chunk_number)

        allow_superruns = [plugins[target_i].allow_superrun for target_i in targets]
        if is_superrun and sum(allow_superruns) not in [0, len(targets)]:
            raise ValueError(
                f"Cannot mix plugins {targets} that allow superruns with those that do not."
            )
        if not sum(allow_superruns) and is_superrun:
            if targets[0].startswith(TEMP_DATA_TYPE_PREFIX):
                raise ValueError(
                    "When only combining subruns, you can only assign one target, "
                    f"but got {plugins[targets[0]].depends_on}!"
                )
            raise ValueError(f"Plugin {targets} does not allowed superrun!")

        # Get savers/loaders, and meanwhile filter out plugins that do not
        # have to do computation. (their instances will stick around
        # though the .deps attribute of plugins that do)
        loaders = dict()
        loader_plugins = dict()
        savers: ty.Dict[str, strax.Saver] = dict()
        seen = set()
        to_compute = dict()

        def check_cache(target_i):
            """For some target, add loaders, and savers where appropriate."""
            nonlocal plugins, loaders, savers, seen
            if target_i in seen:
                return
            seen.add(target_i)
            target_plugin = plugins[target_i]

            # Can we load this data?
            key = self.key_for(run_id, target_i, chunk_number=chunk_number, combining=combining)
            if chunk_number is not None and target_i in chunk_number:
                _chunk_number = chunk_number[target_i]
            else:
                _chunk_number = None
            loader = self._get_partial_loader_for(
                key, time_range=time_range, chunk_number=_chunk_number
            )

            allow_superrun = plugins[target_i].allow_superrun
            if not loader and is_superrun and not allow_superrun or combining:
                # allow_superrun is False so we start to collect the subruns' data_types,
                # which are the depends_on of the superrun's data_type.
                if time_range is not None:
                    raise NotImplementedError("time range loading not yet supported for superruns")

                sub_run_spec = self.run_metadata(run_id, projection="sub_run_spec")["sub_run_spec"]

                # Make subruns if they do not exist.
                self.make(
                    list(sub_run_spec.keys()),
                    target_i,
                    save=(target_i,),
                    multi_run_progress_bar=multi_run_progress_bar,
                    chunk_number=chunk_number,
                )

                ldrs = []
                for subrun in sub_run_spec:
                    # combining is by default False
                    # so we can not combine subrun which is superrun generated in combining mode
                    sub_key = self.key_for(subrun, target_i, chunk_number=chunk_number)

                    if sub_run_spec[subrun] == "all":
                        _subrun_time_range = None
                    else:
                        _subrun_time_range = sub_run_spec[subrun]
                    if chunk_number is not None and target_i in chunk_number:
                        _chunk_number = chunk_number[target_i]
                    else:
                        _chunk_number = None
                    _loader = self._get_partial_loader_for(
                        sub_key, time_range=_subrun_time_range, chunk_number=_chunk_number
                    )
                    if not _loader:
                        raise RuntimeError(
                            f"Could not load {target_i} for subrun {subrun} "
                            "even though we made it? Is the plugin "
                            "you are requesting a SaveWhen.NEVER-plguin?"
                        )
                    ldrs.append(_loader)

                def concat_loader(*args, **kwargs):
                    for x in ldrs:
                        yield from x(*args, **kwargs)

                # pylint: disable=unnecessary-lambda
                loader = lambda *args, **kwargs: concat_loader(*args, **kwargs)

            if loader:
                # Found it! No need to make it or look in other frontends
                loaders[target_i] = loader
                loader_plugins[target_i] = target_plugin
                del plugins[target_i]
            else:
                # Data not found anywhere. We will be computing it.
                self._check_forbidden()
                if (
                    time_range is not None
                    and target_plugin.save_when[target_i] > strax.SaveWhen.EXPLICIT
                ):
                    # While the data type providing the time information is
                    # available (else we'd have failed earlier), one of the
                    # other requested data types is not.
                    error_message = (
                        "Time range selection assumes data is already available,"
                        f" but {target_i} for {run_id} is not."
                    )
                    if target_plugin.save_when[target_i] == strax.SaveWhen.TARGET:
                        error_message += (
                            f"\nFirst run st.make({run_id}, {target_i}) to make {target_i}."
                        )
                    raise strax.DataNotAvailable(error_message)
                if "*" in self.context_config["forbid_creation_of"]:
                    raise strax.DataNotAvailable(
                        f"{target_i} for {run_id} not found in any storage, and "
                        "your context specifies no new data can be created."
                    )
                if target_i in self.context_config["forbid_creation_of"]:
                    raise strax.DataNotAvailable(
                        f"{target_i} for {run_id} not found in any storage, and "
                        "your context specifies it cannot be created."
                    )

                to_compute[target_i] = target_plugin
                for dep_d in target_plugin.depends_on:
                    check_cache(dep_d)

            # In case we can load the data already we want make a new superrun.
            if loader and not (is_superrun and self.context_config["write_superruns"]):
                return

            # Now we should check whether we meet the saving requirements.
            current_plugin_to_savers = [target_i]
            if not self._target_should_be_saved(target_plugin, target_i, targets, save):
                if target_plugin.multi_output:
                    # In case the plugin has more than a single provides we also have to check
                    # whether any of the other data_types should be stored. Hence only remove
                    # the current traget from the list of plugins_to_savers.
                    current_plugin_to_savers = []
                else:
                    # In case of a single-provide plugin we can return now.
                    return

            # Warn about conditions that preclude saving, but the user
            # might not expect.
            if time_range is not None:
                # We're not even getting the whole data.
                self.log.warning(f"Not saving {target_i} while selecting a time range in the run")
                return
            if any([len(v) > 0 for k, v in self._find_options.items() if "fuzzy" in k]):
                # In fuzzy matching mode, we cannot (yet) derive the
                # lineage of any data we are creating. To avoid creating
                # false data entries, we currently do not save at all.
                self.log.warning(f"Not saving {target_i} while fuzzy matching is turned on.")
                return
            if self.context_config["allow_incomplete"]:
                self.log.warning(f"Not saving {target_i} while loading incomplete data is allowed.")
                return

            # Save the target and any other outputs of the plugin.
            if not combining:
                data_type_to_save = set(current_plugin_to_savers + list(target_plugin.provides))
            else:
                # In case of a combining mode we are only interested in the specified target
                data_type_to_save = set(current_plugin_to_savers)
            for d_to_save in data_type_to_save:
                key = self.key_for(
                    run_id, d_to_save, chunk_number=chunk_number, combining=combining
                )
                # Here we just check the availability of key,
                # chunk_number for _get_partial_loader_for can be None
                if self._get_partial_loader_for(key, time_range=time_range):
                    continue

                if not self._target_should_be_saved(
                    target_plugin, d_to_save, targets, save
                ) or savers.get(d_to_save):
                    # This multi-output plugin was scanned before
                    # let's not create doubled savers or store data_types we do not want to.
                    assert target_plugin.multi_output
                    continue

                savers = self._add_saver(
                    savers, d_to_save, run_id, target_plugin, combining=combining
                )

        for target_i in targets:
            check_cache(target_i)
        plugins = to_compute

        intersec = list(plugins.keys() & loaders.keys())
        if len(intersec):
            raise RuntimeError(f"{intersec} both computed and loaded?!")
        if len(targets) > 1:
            pendants = set(targets) & set(self._get_end_targets(plugins))
            final_plugin = tuple(pendants - set(loaders))[:1]
            self.log.warning(
                "Multiple targets detected! This is only suitable for mass "
                f"producing dataypes since only {final_plugin} will be "
                "subscribed in the mailbox system!"
            )
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
            targets=strax.to_str_tuple(final_plugin),
        )

    def get_data_key(self, run_id, target, lineage, combining=False):
        """Get datakey for a given run_id, target and lineage.

        If super is detected, the subruns information are added to the key.

        """
        if run_id.startswith("_"):
            sub_run_spec = self.run_metadata(run_id, projection=["sub_run_spec"])["sub_run_spec"]
        else:
            sub_run_spec = None
        return strax.DataKey(run_id, target, lineage, subruns=sub_run_spec, combining=combining)

    def _add_saver(
        self,
        savers: dict,
        d_to_save: str,
        run_id,
        target_plugin: strax.Plugin,
        combining: bool = False,
    ):
        """Adds savers to already existing savers. Checks if data_type can be stored in any storage
        frontend.

        :param savers: Dictionary of already existing savers.
        :param d_to_save: String of the data_type to be saved.
        :param target_plugin: Plugin which produces the data_type
        :return: Updated savers dictionary.

        """
        key = self.get_data_key(run_id, d_to_save, target_plugin.lineage, combining=combining)
        for sf in self._sorted_storage:
            if sf.readonly:
                continue
            # If we get here, we must try to save
            try:
                saver = sf.saver(
                    key,
                    metadata=target_plugin.metadata(run_id, d_to_save),
                    saver_timeout=self.context_config["saver_timeout"],
                )
                # Now that we are surely saving, make an entry in savers
                savers.setdefault(d_to_save, [])
                savers[d_to_save].append(saver)
            except strax.DataNotAvailable:
                # This frontend cannot save. Too bad.
                pass
        return savers

    @staticmethod
    def _target_should_be_saved(target_plugin, target, targets, save):
        """Function which checks if a given target should be saved.

        :param target_plugin: Plugin to compute target data_type.
        :param target: Target data_type.
        :param targets: Other targets to be computed.
        :param save: Targets to be saved.

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
        return True

    def estimate_run_start_and_end(self, run_id, targets=None):
        """Return run start and end time in ns since epoch.

        This fetches from run metadata, and if this fails, it estimates it using data metadata from
        the targets or the underlying data-types (if it is stored).

        """
        try:
            res = []
            for i in ("start", "end"):
                # Use run metadata, if it is available, to get
                # the run start time (floored to seconds)
                t = self.run_metadata(run_id, i)[i]
                t = t.replace(tzinfo=datetime.timezone.utc)
                t = int(t.timestamp()) * int(1e9)
                res.append(t)
            return res
        except (strax.RunMetadataNotAvailable, KeyError) as e:
            self.log.debug(f"Could not infer start/stop due to type {type(e)} {e}")
            pass
        # Get an approx start from the data itself,
        # then floor it to seconds for consistency
        if targets:
            self.log.debug("Infer start/stop from targets")
            for t in self._get_plugins(
                strax.to_str_tuple(targets),
                run_id,
            ).keys():
                if not self.is_stored(run_id, t):
                    continue
                self.log.debug(f"Try inferring start/stop from {t}")
                try:
                    t0 = self.get_metadata(run_id, t)["chunks"][0]["start"]
                    t0 = (int(t0) // int(1e9)) * int(1e9)

                    t1 = self.get_metadata(run_id, t)["chunks"][-1]["end"]
                    t1 = (int(t1) // int(1e9)) * int(1e9)
                    return t0, t1
                except strax.DataNotAvailable:
                    pass
        self.log.warning(
            "Could not estimate run start and end time from run metadata: assuming it is 0 and inf"
        )
        return 0, float("inf")

    def to_absolute_time_range(
        self,
        run_id,
        targets=None,
        time_range=None,
        seconds_range=None,
        time_within=None,
        full_range=None,
    ):
        """Return (start, stop) time in ns since unix epoch corresponding to time range.

        :param run_id: run id to get
        :param time_range: (start, stop) time in ns since unix epoch. Will be returned without
            modification
        :param targets: data types. Used only if run metadata is unavailable, so run start time has
            to be estimated from data.
        :param seconds_range: (start, stop) seconds since start of run
        :param time_within: row of strax data (e.g. eent)
        :param full_range: If True returns full time_range of the run.

        """

        selection = (
            (time_range is None)
            + (seconds_range is None)
            + (time_within is None)
            + (full_range is None)
        )
        if selection < 2:
            raise RuntimeError(
                "Pass no more than one one of time_range, seconds_range, time_within, or full_range"
            )
        if seconds_range is not None:
            t0, _ = self.estimate_run_start_and_end(run_id, targets)
            time_range = (t0 + int(1e9 * seconds_range[0]), t0 + int(1e9 * seconds_range[1]))
        if time_within is not None:
            time_range = (time_within["time"], strax.endtime(time_within))
        if time_range is not None:
            # Force time range to be integers, since float math on large numbers
            # in not precise
            time_range = tuple([int(x) for x in time_range])

        if full_range:
            time_range = self.estimate_run_start_and_end(run_id, targets)
        return time_range

    def get_iter(
        self,
        run_id: str,
        targets,
        save=tuple(),
        max_workers=None,
        time_range=None,
        seconds_range=None,
        time_within=None,
        time_selection="fully_contained",
        selection=None,
        keep_columns=None,
        drop_columns=None,
        allow_multiple=False,
        progress_bar=True,
        multi_run_progress_bar=True,
        chunk_number=None,
        processor=None,
        combining=False,
        **kwargs,
    ) -> ty.Iterator[strax.Chunk]:
        """Compute target for run_id and iterate over results.

        Do NOT interrupt the iterator (i.e. break): it will keep running stuff
        in background threads...
        {get_docs}

        """
        if hasattr(run_id, "decode"):
            # Byte string has to be decoded:
            run_id = run_id.decode("utf-8")

        # If any new options given, replace the current context
        # with a temporary one
        if len(kwargs):
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        # Convert alternate time arguments to absolute range
        time_range = self.to_absolute_time_range(
            run_id=run_id,
            targets=targets,
            time_range=time_range,
            seconds_range=seconds_range,
            time_within=time_within,
        )

        # Keep a copy of the list of targets for apply_function
        # (otherwise potentially overwritten in temp-plugin)
        targets_list = targets

        if processor is None:
            processor = list(self.processors)[0]

        if isinstance(processor, str):
            processor = self.processors[processor]

        if not hasattr(processor, "iter"):
            raise ValueError("Processors must implement a iter methed.")

        is_superrun = run_id.startswith("_")

        # If multiple targets of the same kind, create a MergeOnlyPlugin
        # to merge the results automatically.
        if isinstance(targets, (list, tuple)) and len(targets) > 1:
            targets = tuple(set(strax.to_str_tuple(targets)))
            plugins = self._get_plugins(targets=targets, run_id=run_id, chunk_number=chunk_number)
            if len(set(plugins[d].data_kind_for(d) for d in targets)) == 1:
                temp_name = TEMP_DATA_TYPE_PREFIX + strax.deterministic_hash(targets)
                p = type(temp_name, (strax.MergeOnlyPlugin,), dict(depends_on=tuple(targets)))
                self.register(p)
                targets = (temp_name,)
            elif not allow_multiple or processor is strax.SingleThreadProcessor:
                raise RuntimeError("Cannot automerge different data kinds!")
            elif self.context_config["timeout"] > 7200 or (
                self.context_config["allow_lazy"] and not self.context_config["allow_multiprocess"]
            ):
                # For allow_multiple we don't want allow this when in lazy mode
                # with long timeouts (lazy-mode is disabled if multiprocessing
                # so if that is activated, we can also continue)
                raise RuntimeError(f"Cannot allow_multiple in lazy mode or with long timeouts.")

        components = self.get_components(
            run_id,
            targets=targets,
            save=save,
            time_range=time_range,
            chunk_number=chunk_number,
            multi_run_progress_bar=multi_run_progress_bar,
            combining=combining,
        )

        # Cleanup the temp plugins
        for k in list(self._plugin_class_registry.keys()):
            if k.startswith("_temp"):
                del self._plugin_class_registry[k]

        seen_a_chunk = False
        generator = processor(
            components,
            max_workers=max_workers,
            allow_shm=self.context_config["allow_shm"],
            allow_multiprocess=self.context_config["allow_multiprocess"],
            allow_rechunk=self.context_config["allow_rechunk"],
            allow_lazy=self.context_config["allow_lazy"],
            max_messages=self.context_config["max_messages"],
            timeout=self.context_config["timeout"],
            is_superrun=is_superrun,
        ).iter()

        try:
            _p, t_start, t_end = self._make_progress_bar(
                run_id, targets=targets, progress_bar=progress_bar
            )
            with _p as pbar:
                pbar.strax_print_time = time.perf_counter()
                pbar.mbs = []
                for n_chunks, result in enumerate(strax.continuity_check(generator), 1):
                    seen_a_chunk = True
                    if not isinstance(result, strax.Chunk):
                        raise ValueError(
                            f"Got type {type(result)} rather than a strax Chunk from the processor!"
                        )
                    # Apply functions known to contexts if any.
                    result.data = self._apply_function(result.data, run_id, targets_list)

                    result.data = strax.apply_selection(
                        result.data,
                        selection=selection,
                        keep_columns=keep_columns,
                        drop_columns=drop_columns,
                        time_range=time_range,
                        time_selection=time_selection,
                    )
                    self._update_progress_bar(
                        pbar, t_start, t_end, n_chunks, result.end, result.nbytes
                    )
                    pbar.strax_print_time = time.perf_counter()
                    yield result
            _p.close()

        except GeneratorExit:
            generator.throw(
                OutsideException(
                    "Terminating due to an exception originating from outside "
                    "strax's get_iter (which we cannot retrieve)."
                )
            )

        except Exception as e:
            generator.throw(e)
            raise ValueError(f"Failed to process chunk {n_chunks}!")

        if not seen_a_chunk:
            if time_range is None:
                raise strax.DataCorrupted("No data returned!")
            raise ValueError(f"Invalid time range: {time_range}, returned no chunks!")

    def _make_progress_bar(self, run_id, targets, progress_bar=True):
        """Make a progress bar for get_iter.

        :param run_id, targets: run_id and targets
        :param progress_bar: Bool whether or not to display the progress bar
        :return: progress bar, t_start (run) and t_end (run)

        """
        try:
            (
                t_start,
                t_end,
            ) = self.estimate_run_start_and_end(run_id, targets)
        except (AttributeError, KeyError, IndexError):
            # During testing some thing remain a secret
            (
                t_start,
                t_end,
            ) = 0, float("inf")
        if t_end == float("inf"):
            progress_bar = False

        # Define nice progressbar format:
        bar_format = (
            "{desc}: "  # The loading plugin x
            "|{bar}| "  # Bar that is being filled
            "{percentage:.2f} % "  # Percentage
            "[{elapsed}<{remaining}]"  # Time estimate
            "{postfix}"  # Extra info
        )
        description = f'Loading {"plugins" if targets[0].startswith("_temp") else targets}'
        pbar = tqdm(
            total=1, desc=description, bar_format=bar_format, leave=True, disable=not progress_bar
        )
        return pbar, t_start, t_end

    @staticmethod
    def _update_progress_bar(pbar, t_start, t_end, n_chunks, chunk_end, nbytes):
        """Do some tqdm voodoo to get the progress bar for st.get_iter."""
        if t_end - t_start > 0:
            fraction_done = (chunk_end - t_start) / (t_end - t_start)
            if fraction_done > 0.99:
                # Patch to 1 to not have a red pbar when very close to 100%
                fraction_done = 1
            pbar.n = np.clip(fraction_done, 0, 1)
        else:
            # Strange, start and endtime are the same, probably we don't
            # have data yet e.g. allow_incomplete == True.
            pbar.n = 0
        # Let's add the postfix which is the info behind the tqdm marker
        seconds_per_chunk = time.perf_counter() - pbar.strax_print_time
        pbar.mbs.append((nbytes / 1e6) / seconds_per_chunk)
        mbs = np.mean(pbar.mbs)
        if mbs < 1:
            rate = f"{mbs * 1000:.1f} kB/s"
        else:
            rate = f"{mbs:.1f} MB/s"
        postfix = f"#{n_chunks} ({seconds_per_chunk:.2f} s). {rate}"
        pbar.set_postfix_str(postfix)
        pbar.update(0)

    def make(
        self,
        run_id: ty.Union[str, tuple, list],
        targets,
        save=tuple(),
        max_workers=None,
        _skip_if_built=True,
        chunk_number=None,
        combining=False,
        **kwargs,
    ) -> None:
        """Compute target for run_id. Returns nothing (None).

        {get_docs}

        """
        kwargs.setdefault("progress_bar", False)

        # Multi-run support
        run_ids = strax.to_str_tuple(run_id)
        if len(run_ids) == 0:
            raise ValueError("Cannot build empty list of runs")
        if len(run_ids) > 1:
            return strax.multi_run(
                self.get_array,
                run_ids,
                targets=targets,
                throw_away_result=True,
                log=self.log,
                save=save,
                max_workers=max_workers,
                chunk_number=chunk_number,
                **kwargs,
            )

        if _skip_if_built and self.is_stored(
            run_ids[0], targets, chunk_number=chunk_number, combining=combining
        ):
            return

        for _ in self.get_iter(
            run_ids[0],
            targets,
            save=save,
            max_workers=max_workers,
            chunk_number=chunk_number,
            combining=combining,
            **kwargs,
        ):
            pass

    def get_array(
        self, run_id: ty.Union[str, tuple, list], targets, save=tuple(), max_workers=None, **kwargs
    ) -> np.ndarray:
        """Compute target for run_id and return as numpy array.

        {get_docs}

        """
        run_ids = strax.to_str_tuple(run_id)

        if kwargs.get("allow_multiple", False):
            raise RuntimeError("Cannot allow_multiple with get_array/get_df")

        if len(run_ids) > 1:
            results = strax.multi_run(
                self.get_array,
                run_ids,
                targets=targets,
                log=self.log,
                save=save,
                max_workers=max_workers,
                **kwargs,
            )
        else:
            source = self.get_iter(
                run_ids[0], targets, save=save, max_workers=max_workers, **kwargs
            )
            results = [x.data for x in source]

        results = np.concatenate(results)
        return results

    def accumulate(
        self,
        run_id: str,
        targets: ty.Union[ty.Tuple[str], ty.List[str]],
        fields=None,
        function=None,
        store_first_for_others=True,
        function_takes_fields=False,
        **kwargs,
    ):
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

        :return dictionary: Dictionary with the accumulated result;
            see function and store_first_for_others arguments.
            Four fields are always added:
                start: start time of the first processed chunk
                end: end time of the last processed chunk
                n_chunks: number of chunks in run
                n_rows: number of data entries in run

        """
        if kwargs.get("allow_multiple", False):
            raise RuntimeError("Cannot allow_multiple with accumulate")

        n_chunks = 0
        seen_data = False
        result = {"n_rows": 0}
        if fields is not None:
            fields = strax.to_str_tuple(fields)
        if function is None:

            def function(arr):
                return arr

            function_takes_fields = False

        for chunk in self.get_iter(run_id, targets, **kwargs):
            data = chunk.data

            if n_chunks == 0:
                result["start"] = chunk.start
                if fields is None:
                    # Sum all fields except time and endtime
                    fields = [x for x in data.dtype.names if x not in ("time", "endtime")]

            if store_first_for_others and not seen_data and len(data):
                # Store the first value we see for the non-accumulated fields
                for name in data.dtype.names:
                    if name not in fields:
                        result[name] = data[0][name]
                seen_data = True
            result["end"] = chunk.end
            result["n_rows"] += len(data)

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

            elif isinstance(data, dict) or (
                isinstance(data, np.ndarray) and data.dtype.fields is not None
            ):
                # Function returned record array or dict
                for field in fields:
                    result[field] = result.get(field, 0) + np.sum(data[field], axis=0)
            else:
                # Function returned a scalar or flat array
                result["result"] = np.sum(data, axis=0) + result.get("result", 0)
            n_chunks += 1
        result["n_chunks"] = n_chunks
        return result

    def get_df(
        self, run_id: ty.Union[str, tuple, list], targets, save=tuple(), max_workers=None, **kwargs
    ) -> pd.DataFrame:
        """Compute target for run_id and return as pandas DataFrame.

        {get_docs}

        """
        df = self.get_array(run_id, targets, save=save, max_workers=max_workers, **kwargs)

        return strax.convert_structured_array_to_df(df, log=self.log)

    def get_zarr(
        self,
        run_ids,
        targets,
        storage="./strax_temp_data",
        progress_bar=False,
        overwrite=True,
        **kwargs,
    ):
        """Get persistent arrays using zarr. This is useful when loading large amounts of data that
        cannot fit in memory zarr is very compatible with dask. Targets are loaded into separate
        arrays and runs are merged. the data is added to any existing data in the storage location.

        :param run_ids: (Iterable) run ids you wish to load.
        :param targets: (Iterable) targets to load.
        :param storage: (str, optional) fsspec path to store array. Defaults to './strax_temp_data'.
        :param overwrite: (boolean, optional) whether to overwrite existing arrays for targets at
            given path.
        :return zarr.Group: zarr group containing the persistant arrays available at the storage
            location after loading the requested data the runs loaded into a given array can be seen
            in the array .attrs['RUNS'] field

        """
        import zarr

        context_hash = self._context_hash()
        kwargs_hash = strax.deterministic_hash(kwargs)
        root = zarr.open(storage, mode="w")
        group = root.require_group(context_hash + "/" + kwargs_hash, overwrite=overwrite)
        for target in strax.to_str_tuple(targets):
            idx = 0
            zarray = None
            if target in group:
                zarray = group[target]
                if not overwrite:
                    idx = zarray.size
            INSERTED = {}
            for run_id in strax.to_str_tuple(run_ids):
                if zarray is not None and run_id in zarray.attrs.get("RUNS", {}):
                    continue
                key = self.key_for(
                    run_id,
                    target,
                    chunk_number=kwargs.get("chunk_number", None),
                    combining=kwargs.get("combining", False),
                )
                INSERTED[run_id] = dict(start_idx=idx, end_idx=idx, lineage_hash=key.lineage_hash)
                for chunk in self.get_iter(run_id, target, progress_bar=progress_bar, **kwargs):
                    end_idx = idx + chunk.data.size
                    if zarray is None:
                        dtype = [(d[0][1],) + d[1:] for d in chunk.dtype.descr]
                        zarray = group.create_dataset(target, shape=end_idx, dtype=dtype)
                    else:
                        zarray.resize(end_idx)
                    zarray[idx:end_idx] = chunk.data
                    idx = end_idx
                    INSERTED[run_id]["end_idx"] = end_idx
            zarray.attrs["RUNS"] = dict(zarray.attrs.get("RUNS", {}), **INSERTED)
        return group

    def key_for(self, run_id, target, chunk_number=None, combining=False):
        """Get the DataKey for a given run and a given target plugin. The DataKey is inferred from
        the plugin lineage. The lineage can come either from the _fixed_plugin_cache or computed on
        the fly.

        :param run_id: run id to get
        :param target: data type to get
        :return: strax.DataKey of the target

        """
        if self._plugins_are_cached((target,), chunk_number=chunk_number):
            context_hash = self._context_hash()
            if context_hash in self._fixed_plugin_cache:
                plugins = self._fixed_plugin_cache[self._context_hash()]
            else:
                # This once happened due to temp. plugins, should not happen again
                self.log.warning(
                    f"Context hash changed to {context_hash} for {self._plugin_class_registry}?"
                )
                plugins = self._get_plugins((target,), run_id, chunk_number=chunk_number)
        else:
            plugins = self._get_plugins((target,), run_id, chunk_number=chunk_number)

        lineage = plugins[target].lineage
        return self.get_data_key(run_id, target, lineage, combining=combining)

    def get_metadata(self, run_id, target, chunk_number=None, combining=False) -> dict:
        """Return metadata for target for run_id, or raise DataNotAvailable if data is not yet
        available.

        :param run_id: run id to get
        :param target: data type to get

        """
        key = self.key_for(run_id, target, chunk_number=chunk_number, combining=combining)
        for sf in self._sorted_storage:
            try:
                return sf.get_metadata(key, **self._find_options)
            except strax.DataNotAvailable:
                self.log.debug(f"Frontend {sf} does not have {key}")
        raise strax.DataNotAvailable(f"Can't load metadata, data for {key} not available")

    def compare_metadata(self, data1, data2, return_results=False):
        """Compare the metadata between two strax data.

        :param data1, data2: either a list (tuple) of runid + target pair, or path to metadata to
        compare,     or a dictionary of the metadata
        :param return_results: bool, if True, returns a dictionary with metadata and lineages that
            are found for the inputs does not do the comparison

        example usage:
            context.compare_metadata(("053877", "peak_basics"), "./my_path_to/JSONfile.json")
            first_metadata = context.get_metadata(run_id, "events")
            context.compare_metadata(
                 ("053877", "peak_basics"), first_metadata)
            context.compare_metadata(
                ("053877", "records"), ("053899", "records") )
            results_dict = context.compare_metadata(
                ("053877", "peak_basics"), ("053877", "events_info"),
                 return_results=True)

        """

        def _extract_input(data):
            """Identify and extract the given input.

            User can either pass a `runid + target` pair or path to metadata json file or a dict

            """
            if isinstance(data, (tuple, list)):
                run_id, target = data
                metafile = None
            elif isinstance(data, str):
                run_id, target = None, None
                metafile = data
            elif isinstance(data, dict):
                run_id, target = None, None
                metafile = data
            else:
                raise ValueError(
                    "data can either be a tuple(list) with runid+target or the path of the metadata"
                    " json file"
                )
            return run_id, target, metafile

        def _extract_metadata_and_lineage(self, run_id, target, metafile):
            """Extract the actual metadata and lineage based on given inputs and whether the data is
            available."""
            # if the runid+target pair is given, check if stored
            if metafile is None:
                _is_stored = self.is_stored(run_id, target)
                metadata = self.get_metadata(run_id, target) if _is_stored else None
                lineage = (
                    metadata["lineage"] if _is_stored else self.key_for(run_id, target).lineage
                )
            elif isinstance(metafile, dict):
                metadata = metafile
                lineage = metadata["lineage"]
            else:
                # metafile is given instead of run id + target pair
                with open(metafile) as json_file:
                    metadata = json.load(json_file)
                    lineage = metadata["lineage"]
            # streamline all lineages
            lineage = strax.utils.convert_tuple_to_list(lineage)
            return metadata, lineage

        run_id1, target1, metafile1 = _extract_input(data1)
        run_id2, target2, metafile2 = _extract_input(data2)

        metadata1, lineage1 = _extract_metadata_and_lineage(self, run_id1, target1, metafile1)
        metadata2, lineage2 = _extract_metadata_and_lineage(self, run_id2, target2, metafile2)

        if return_results:
            results_dict = {
                "metadata1": metadata1,
                "lineage1": lineage1,
                "metadata2": metadata2,
                "lineage2": lineage2,
            }
            print(
                f"Returning the collected data, dictionaries can be compared by"
                f" `strax.utils.compare_dict(d1,d2)`"
            )
            return results_dict

        # if both metadata exists, simple comparison
        # do a full metadata comparison if the whole metadata exists
        if metadata1 is not None and metadata2 is not None:
            print("Both metadata exists!")
            strax.utils.compare_dict(metadata1, metadata2)
        else:
            print(f"Both metadata are not available together. Comparing lineages!")
            strax.utils.compare_dict(lineage1, lineage2)

    def run_metadata(self, run_id, projection=None) -> dict:
        """Return run-level metadata for run_id, or raise DataNotAvailable if this is not available.

        :param run_id: run id to get
        :param projection: Selection of fields to get, following MongoDB syntax. May not be
            supported by frontend.

        """
        for sf in self._sorted_storage:
            if not sf.provide_run_metadata:
                continue
            try:
                return sf.run_metadata(run_id, projection=projection)
            except (strax.DataNotAvailable, NotImplementedError):
                self.log.debug(f"Frontend {sf} does not have run metadata for {run_id}")
        raise strax.RunMetadataNotAvailable(f"No run-level metadata available for {run_id}")

    def size_mb(self, run_id, target):
        """Return megabytes of memory required to hold data."""
        md = self.get_metadata(run_id, target)
        return sum([x["nbytes"] for x in md["chunks"]]) / 1e6

    def run_defaults(self, run_id):
        """Get configuration defaults from the run metadata (if these exist)

        This will only call the rundb once for each run while the context is in existence; further
        calls to this will return a cached value.

        """
        if not self.context_config["use_per_run_defaults"]:
            return dict()
        if run_id in self._run_defaults_cache:
            return self._run_defaults_cache[run_id]
        try:
            defs = self.run_metadata(run_id, projection=RUN_DEFAULTS_KEY).get(
                RUN_DEFAULTS_KEY, dict()
            )
        except strax.RunMetadataNotAvailable:
            defs = dict()
        self._run_defaults_cache[run_id] = defs
        return defs

    def is_stored(
        self, run_id, target, detailed=False, chunk_number=None, combining=False, **kwargs
    ):
        """Return whether data type target has been saved for run_id through any of the registered
        storage frontends.

        Note that even if False is returned, the data type may still be made with a trivial
        computation.

        """
        if isinstance(target, (tuple, list)):
            return all(
                [
                    self.is_stored(
                        run_id, t, chunk_number=chunk_number, combining=combining, **kwargs
                    )
                    for t in target
                ]
            )

        # If any new options given, replace the current context
        # with a temporary one
        if len(kwargs):
            # Comment below disables pycharm from inspecting the line below it
            # noinspection PyMethodFirstArgAssignment
            self = self.new_context(**kwargs)

        for sf in self._sorted_storage:
            if self._is_stored_in_sf(
                run_id, target, sf, chunk_number=chunk_number, combining=combining
            ):
                return True
        # None of the frontends has the data

        # Before returning False, check if the data can be made trivially
        plugin = self._plugin_class_registry[target]
        save_when = plugin.save_when

        # Mutli-target plugins provide a save_when per target
        if isinstance(save_when, immutabledict):
            save_when = save_when[target]

        if save_when < strax.SaveWhen.ALWAYS and detailed:
            self.log.warning(
                f"{target} is not set to always be saved. "
                "This is probably because it can be trivially made from other data. "
                f"{target} depends on {plugin.depends_on}. Check if these are stored."
            )

        return False

    def _check_forbidden(self):
        """Check that the forbid_creation_of config is of tuple type.

        Otherwise, try to make it a tuple

        """
        self.context_config["forbid_creation_of"] = strax.to_str_tuple(
            self.context_config["forbid_creation_of"]
        )

    def _apply_function(
        self,
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
        apply_functions = self.context_config["apply_data_function"]
        if hasattr(apply_functions, "__call__"):
            # Apparently someone did not read the docstring and inserted
            # a single function instead of a list.
            apply_functions = [apply_functions]
        if not isinstance(apply_functions, (tuple, list)):
            raise ValueError(
                "apply_data_function in context config should be tuple of "
                f"functions. Instead got {apply_functions}"
            )
        for function in apply_functions:
            if not hasattr(function, "__call__"):
                raise TypeError(
                    "apply_data_function in the context_config got "
                    f"{function} but expected callable function with two "
                    "positional arguments: f(data, targets)."
                )
            # Make sure that the function takes two arguments (data and targets)
            chunk_data = function(chunk_data, run_id, targets)
        return chunk_data

    def _check_copy_to_frontend_kwargs(
        self, run_id, target, target_frontend_id, rechunk, rechunk_to_mb
    ) -> None:
        """Simple kwargs checks for copy_to_frontend."""
        if not self.is_stored(run_id, target):
            raise strax.DataNotAvailable(f"Cannot copy {run_id} {target} since it does not exist")
        if rechunk and rechunk_to_mb == strax.DEFAULT_CHUNK_SIZE_MB:
            self.log.warning("No <rechunk_to_mb> specified!")
        # Reuse some codes
        self._check_merge_per_chunk_storage_kwargs(run_id, target, target_frontend_id)

    def _get_target_sf(self, run_id, target, target_frontend_id):
        """Get the target storage frontends for copy_to_frontend and merge_per_chunk_storage."""
        if target_frontend_id is None:
            target_sf = self.storage
        elif len(self.storage) > target_frontend_id:
            target_sf = [self.storage[target_frontend_id]]

        # Keep frontends that:
        #  1. don't already have the data; and
        #  2. take the data; and
        #  3. are not readonly
        target_sf = [
            t_sf
            for t_sf in target_sf
            if (
                not self._is_stored_in_sf(run_id, target, t_sf)
                and t_sf._we_take(target)
                and t_sf.readonly is False
            )
        ]
        return target_sf

    def copy_to_frontend(
        self,
        run_id: str,
        target: str,
        target_frontend_id: ty.Optional[int] = None,
        target_compressor: ty.Optional[str] = None,
        rechunk: bool = False,
        rechunk_to_mb: int = strax.DEFAULT_CHUNK_SIZE_MB,
    ):
        """Copy data from one frontend to another.

        :param run_id: run_id
        :param target: target datakind
        :param target_frontend_id: index of the frontend that the data should go to in
            context.storage. If no index is specified, try all.
        :param target_compressor: if specified, recompress with this compressor.
        :param rechunk: allow re-chunking for saving
        :param rechunk_to_mb: rechunk to specified target size. Only works if rechunk is True.

        """
        # NB! We don't want to use self._sorted_storage here since the order matters!

        self._check_copy_to_frontend_kwargs(
            run_id, target, target_frontend_id, rechunk, rechunk_to_mb
        )

        # Figure out which of the frontends has the data. Raise error when none
        source_sf = self.get_source_sf(run_id, target, should_exist=True)[0]

        # Get the target storage frontends
        target_sf = self._get_target_sf(run_id, target, target_frontend_id)
        self.log.info(f"Copy data from {source_sf} to {target_sf}")

        if not len(target_sf):
            raise ValueError(
                "No frontend to copy to! Perhaps you already stored "
                "it or none of the frontends is willing to take it?"
            )

        # Get the info from the source backend (s_be) that we need to fill
        # the target backend (t_be) with
        data_key = self.key_for(run_id, target)
        # This should never fail, we just tried
        s_be_str, s_be_key = source_sf.find(data_key)
        s_be = source_sf._get_backend(s_be_str)
        md = s_be.get_metadata(s_be_key)

        if target_compressor is not None:
            self.log.info(f'Changing compressor {md["compressor"]} -> {target_compressor}.')
            md.update({"compressor": target_compressor})

        if rechunk and md["chunk_target_size_mb"] != rechunk_to_mb:
            self.log.info(f'Changing chunk-size: {md["chunk_target_size_mb"]} -> {rechunk_to_mb}.')
            md.update({"chunk_target_size_mb": rechunk_to_mb})

        for t_sf in target_sf:
            try:
                # Need to load a new loader each time since it's a generator
                # and will be exhausted otherwise.
                loader = s_be.loader(s_be_key)

                def wrapped_loader():
                    """Wrapped loader for changing the target_size_mb."""
                    while True:
                        try:
                            # pylint: disable=cell-var-from-loop
                            data = next(loader)
                            # Update target chunk size for re-chunking
                            data.target_size_mb = md["chunk_target_size_mb"]
                        except StopIteration:
                            return
                        yield data

                # Fill the target buffer
                t_be_str, t_be_key = t_sf.find(data_key, write=True)
                target_be = t_sf._get_backend(t_be_str)
                saver = target_be._saver(t_be_key, md)
                saver.save_from(wrapped_loader(), rechunk=rechunk)
            except NotImplementedError:
                # Target is not susceptible
                continue
            except strax.DataExistsError:
                raise strax.DataExistsError(
                    f"Trying to write {data_key} to {t_sf} which already exists, "
                    "do you have two storage frontends writing to the same place?"
                )

    def _check_merge_per_chunk_storage_kwargs(self, run_id, target, target_frontend_id) -> None:
        if len(strax.to_str_tuple(target)) > 1:
            raise ValueError("copy_to_frontend only works for a single target at the time")
        if target_frontend_id is not None and target_frontend_id >= len(self.storage):
            raise ValueError(
                f"Cannot select {target_frontend_id}-th frontend as "
                f"we only have {len(self.storage)} frontends!"
            )

    def merge_per_chunk_storage(
        self,
        run_id: str,
        target: str,
        per_chunked_dependency: str,
        rechunk=True,
        chunk_number_group: ty.Optional[ty.List[ty.List[int]]] = None,
        target_frontend_id: ty.Optional[int] = None,
        check_is_stored: bool = True,
    ):
        """Merge the per-chunked data from the per-chunked dependency into the target storage."""

        if check_is_stored and self.is_stored(run_id, target):
            raise ValueError(f"Data {target} for {run_id} already exists.")

        self._check_merge_per_chunk_storage_kwargs(run_id, target, target_frontend_id)

        chunks = self.get_metadata(run_id, per_chunked_dependency)["chunks"]
        if chunk_number_group is not None:
            combined_chunk_numbers = list(itertools.chain.from_iterable(chunk_number_group))
            if len(combined_chunk_numbers) != len(set(combined_chunk_numbers)):
                raise ValueError(f"Duplicate chunk numbers found in {chunk_number_group}")
            if min(combined_chunk_numbers) == 0 and max(combined_chunk_numbers) == len(chunks) - 1:
                # If the chunks are all the chunks of the dependency, we can drop the chunk_number
                _chunk_number = None
            else:
                _chunk_number = {per_chunked_dependency: combined_chunk_numbers}
        else:
            # if no chunk numbers are given, use information from the dependency
            chunk_number_group = [[c["chunk_i"]] for c in chunks]
            _chunk_number = None

        # Make sure that all needed runs are stored
        for chunk_number in chunk_number_group:
            assert self.is_stored(
                run_id, target, chunk_number={per_chunked_dependency: chunk_number}
            )

        # Usually we want to save in the same storage frontend
        # Here we assume that the target is stored chunk by chunk of the dependency
        source_sf = self.get_source_sf(
            run_id,
            target,
            chunk_number={per_chunked_dependency: chunk_number},
            should_exist=True,
        )[0]

        # Get the target storage frontends
        target_sf = self._get_target_sf(run_id, target, target_frontend_id)

        def wrapped_loader():
            """Wrapped loader for changing the target_size_mb."""
            for chunk_number in chunk_number_group:
                # Mostly revised from self.copy_to_frontend
                # Get the info from the source backend (s_be) that we need to fill
                # the target backend (t_be) with
                data_key = self.key_for(
                    run_id, target, chunk_number={per_chunked_dependency: chunk_number}
                )
                # This should never fail, we just tried
                s_be_str, s_be_key = source_sf.find(data_key)
                s_be = source_sf._get_backend(s_be_str)
                md = s_be.get_metadata(s_be_key)

                loader = s_be.loader(s_be_key)
                try:
                    while True:
                        # pylint: disable=cell-var-from-loop
                        data = next(loader)
                        # Update target chunk size for re-chunking
                        data.target_size_mb = md["chunk_target_size_mb"]
                        yield data
                except StopIteration:
                    continue

        data_key = self.key_for(run_id, target, chunk_number=_chunk_number)
        target_plugin = self.__get_plugin(run_id, target, chunk_number=_chunk_number)
        target_md = target_plugin.metadata(run_id, target)
        target_md["strax_version"] = strax.__version__
        # Copied from StorageBackend.saver
        if "dtype" in target_md:
            target_md["dtype"] = target_md["dtype"].descr.__repr__()
        for t_sf in target_sf:
            # Fill the target buffer
            t_be_str, t_be_key = t_sf.find(data_key, write=True)
            target_be = t_sf._get_backend(t_be_str)
            saver = target_be._saver(t_be_key, target_md)
            saver.save_from(wrapped_loader(), rechunk=rechunk)

    def get_source(
        self,
        run_id: str,
        target: str,
        check_forbidden: bool = True,
    ) -> ty.Union[set, None]:
        """For a given run_id and target get the stored bases where we can start processing from, if
        no base is available, return None.

        :param run_id: run_id
        :param target: target
        :param check_forbidden: Check that we are not requesting to make a plugin that is forbidden
            by the context to be created.
        :return: set of plugin names that are needed to start processing from and are needed in
            order to build this target.

        """
        try:
            return set(
                plugin_name
                for plugin_name, plugin_stored in self.stored_dependencies(
                    run_id=run_id, target=target, check_forbidden=check_forbidden
                ).items()
                if plugin_stored
            )
        except strax.DataNotAvailable:
            return None

    def stored_dependencies(
        self,
        run_id: str,
        target: ty.Union[str, list, tuple],
        check_forbidden: bool = True,
        _targets_stored: ty.Optional[dict] = None,
    ) -> ty.Optional[dict]:
        """For a given run_id and target(s) get a dictionary of all the datatypes that are required
        to build the requested target.

        :param run_id: run_id
        :param target: target or a list of targets
        :param check_forbidden: Check that we are not requesting to make a plugin that is forbidden
            by the context to be created.
        :return: dictionary of data types (keys) required for building the requested target(s) and
            if they are stored (values)
        :raises strax.DataNotAvailable: if there is at least one data type that is not stored and
            has no dependency or if it cannot be created

        """
        if _targets_stored is None:
            _targets_stored = dict()

        targets = strax.to_str_tuple(target)
        if len(targets) > 1:
            # Multiple targets, do them all
            for dep in targets:
                self.stored_dependencies(
                    run_id,
                    dep,
                    check_forbidden=check_forbidden,
                    _targets_stored=_targets_stored,
                )
            return _targets_stored

        # Make sure we have the string not ('target',)
        target = targets[0]

        if target in _targets_stored:
            return None

        _targets_stored[target] = self.is_stored(run_id, target)

        if _targets_stored[target]:
            return _targets_stored

        # Need to init the class e.g. if we want to allow depends_on which is not a class attribute
        plugin = self._plugin_class_registry[target]()
        if not plugin.depends_on:
            raise strax.DataNotAvailable(f"Lowest level dependency {target} is not stored")

        forbidden = strax.to_str_tuple(self.context_config["forbid_creation_of"])
        forbidden_warning = (
            "For {run_id}:{target}, you are not allowed to make {dep} and "
            "it is not stored. Disable check with check_forbidden=False"
        )
        if check_forbidden and target in forbidden:
            raise strax.DataNotAvailable(
                forbidden_warning.format(
                    run_id=run_id,
                    target=target,
                    dep=target,
                )
            )

        self.stored_dependencies(
            run_id,
            target=plugin.depends_on,
            check_forbidden=check_forbidden,
            _targets_stored=_targets_stored,
        )
        return _targets_stored

    def _is_stored_in_sf(
        self,
        run_id,
        target,
        storage_frontend: strax.StorageFrontend,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
        combining: bool = False,
    ) -> bool:
        """Check if the storage frontend has the requested datakey for the run_id and target.

        :param storage_frontend: strax.StorageFrontend to check if it has the requested datakey for
            the run_id and target.
        :return: if the frontend has the key or not.

        """
        key = self.key_for(run_id, target, chunk_number=chunk_number, combining=combining)
        try:
            storage_frontend.find(key, **self._find_options)
            return True
        except strax.DataNotAvailable:
            return False

    def get_source_sf(self, run_id, target, should_exist=False, chunk_number=None):
        """Get the source storage frontends for a given run_id and target.

        :param run_id, target: run_id, target
        :param should_exist: Raise a ValueError if we cannot find one (e.g. we already checked the
            data is stored)
        :return: list of strax.StorageFrontend (when should_exist is False)

        """
        if isinstance(target, (tuple, list)):
            if len(target) == 0:
                raise ValueError("Cannot find stored frontend for empty target!")
            frontends_list = [
                self.get_source_sf(
                    run_id,
                    t,
                    should_exist=should_exist,
                    chunk_number=chunk_number,
                )
                for t in target
            ]
            return list(set.intersection(*map(set, frontends_list)))

        frontends = []
        for sf in self._sorted_storage:
            if self._is_stored_in_sf(run_id, target, sf, chunk_number=chunk_number):
                frontends.append(sf)
        if should_exist and not frontends:
            raise ValueError(
                "This cannot happen, we just checked that this run should be stored?!?"
            )
        return frontends

    def get_save_when(self, target: str) -> ty.Union[strax.SaveWhen, int]:
        """For a given plugin, get the save when attribute either being a dict or a number."""
        plugin_class = self._plugin_class_registry[target]
        save_when = plugin_class.save_when
        if isinstance(save_when, immutabledict):
            save_when = save_when[target]
        if not isinstance(save_when, (IntEnum, int)):
            raise ValueError(f"SaveWhen of {plugin_class} should be IntEnum or immutabledict")
        return save_when

    def provided_dtypes(self, runid="0"):
        """Summarize dtype information provided by this context.

        :return: dictionary of provided dtypes with their corresponding lineage hash, save_when,
            version

        """
        hashes = set(
            [
                (
                    data_type,
                    self.key_for(runid, data_type).lineage_hash,
                    self.get_save_when(data_type),
                    plugin.version(),
                )
                for plugin in self._plugin_class_registry.values()
                for data_type in plugin.provides
            ]
        )

        return {
            data_type: dict(hash=_hash, save_when=save_when.name, version=version)
            for data_type, _hash, save_when, version in hashes
        }

    def get_dependency_plugins(
        self,
        target: str,
        run_id: str,
        chunk_number: ty.Optional[ty.Dict[str, ty.List[int]]] = None,
    ) -> ty.Dict[str, strax.Plugin]:
        """Return all plugins required to produce targets.

        :param target: data type to produce
        :param run_id: run id to use for run-dependent config options
        :param chunk_number: Chunk number to use for run-dependent config options
        :return: dictionary with data type as key and plugin as value

        """
        # Get all plugins required to produce targets
        plugin = self.__get_plugin(run_id, target, chunk_number=chunk_number)
        _dependencies = [plugin.deps.items()]
        _dependencies += [
            self.get_dependency_plugins(d, run_id, chunk_number).items() for d in plugin.deps
        ]
        dependencies = dict(itertools.chain.from_iterable(_dependencies))
        return dependencies

    def get_dependencies(self, data_type):
        """Get the dependencies of a data_type."""
        plugin = self._plugin_class_registry[data_type]()
        dependencies = plugin.depends_on
        dependencies += tuple(
            itertools.chain.from_iterable(self.get_dependencies(d) for d in plugin.depends_on)
        )
        return set(dependencies)

    @property
    def root_data_types(self):
        """Root data_type that does not depend on anything."""
        _root_data_types = set()
        for k, v in self._plugin_class_registry.items():
            _v = v()
            if not _v.depends_on:
                _root_data_types |= set((k,))
        return _root_data_types

    @property
    def tree(self):
        """Dependency tree whose key is provides and value is depends_on."""
        _tree = dict()
        for v in self._plugin_class_registry.values():
            _v = v()
            for p in _v.provides:
                _tree.setdefault(p, [])
                _tree[p] += _v.depends_on
        return _tree

    @property
    def inversed_tree(self):
        """Inversed dependency tree whose key is depends_on and value is provides."""
        _inverse_tree = dict()
        for v in self._plugin_class_registry.values():
            _v = v()
            for d in _v.depends_on:
                _inverse_tree.setdefault(d, [])
                _inverse_tree[d] += _v.provides
        return _inverse_tree

    def check_superrun(self):
        """Raise if non-superrun plugins depends on superrun plugins."""
        inversed_tree = self.inversed_tree

        # define a recursive function to check if all the dependencies support superruns
        def check_support_superrun(data_type, checked=set(), seen_allow=None):
            if data_type in checked:
                return checked
            if self._plugin_class_registry[data_type].allow_superrun:
                seen_allow = data_type
            if seen_allow and not self._plugin_class_registry[data_type].allow_superrun:
                raise ValueError(
                    f"Already support superruns in {seen_allow}, "
                    f"but it's dependency {data_type} does not support superruns."
                )
            checked |= set((data_type,))
            if data_type not in inversed_tree:
                return checked
            for d in inversed_tree[data_type]:
                checked |= check_support_superrun(d, checked, seen_allow)
            return checked

        # use checked set to record the data_type that has been checked
        # to shorten the time of checking
        checked = set()
        for data_type in self.root_data_types:
            if self._plugin_class_registry[data_type].allow_superrun:
                seen_allow = data_type
            else:
                seen_allow = None
            checked |= check_support_superrun(data_type, checked, seen_allow)

    @property
    def tree_levels(self):
        """Get the levels of the data types in the context.

        This function will be useful to tell us which data_type to process first.

        For Example, for a given class with Records, Peaks registered, the tree_levels will return:
        {'records': {'level': 0, 'class': 'Records', 'index': 0, 'order': 0}, 'peaks': {'level': 1,
        'class': 'Peaks', 'index': 0, 'order': 1}}

        """

        context_hash = self._context_hash()
        if self._fixed_level_cache is not None and context_hash in self._fixed_level_cache:
            return self._fixed_level_cache[context_hash]

        def _get_levels(data_type=None, results=None):
            """Get the level data_type in the context."""
            if results is None:
                results = dict()
            for k in [data_type] if data_type else self._plugin_class_registry.keys():
                results[k] = dict()
                _v = self._plugin_class_registry[k]()
                if _v.depends_on:
                    results[k]["level"] = (
                        max(_get_levels(d, results)[d]["level"] for d in _v.depends_on) + 1
                    )
                else:
                    results[k]["level"] = 0
                results[k]["class"] = self._plugin_class_registry[k].__name__
                results[k]["index"] = _v.provides.index(k)
            return results

        # Sort the results by level, class, and index in provides
        _results = sorted(
            _get_levels().items(), key=lambda x: (x[1]["level"], x[1]["class"], x[1]["index"])
        )

        # Assign order to the results
        for order, (key, value) in enumerate(_results):
            value["order"] = order
        results = dict(_results)

        if self._fixed_level_cache is None:
            self._fixed_level_cache = {context_hash: results}
        elif context_hash not in self._fixed_level_cache:
            self.log.info("Replacing context._fixed_level_cache since plugins/versions changed")
            self._fixed_level_cache = {context_hash: results}

        return results

    @classmethod
    def add_method(cls, f):
        """Add f as a new Context method."""
        setattr(cls, f.__name__, f)


select_docs = """
:param selection: Query string, sequence of strings, or simple function to apply.
    The function must take a single argument which represents the structure
    numpy array of the loaded data.
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
:param time_selection: Kind of time selection to apply:
    - fully_contained: (default) select things fully contained in the range
    - touching: select things that (partially) overlap with the range
    - skip: Do not select a time range, even if other arguments say so
:param chunk_number: Load chunk from the dependency according to this dictionary.
:param progress_bar: Display a progress bar if metedata exists.
:param multi_run_progress_bar: Display a progress bar for loading multiple runs
"""

get_docs = """
:param run_id: run id to get
:param targets: list/tuple of strings of data type names to get
:param ignore_errors: Return the data for the runs that successfully loaded, even if some runs
        failed executing.
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
:param processor: Name of the processor to use. If not specified, the
    first processor from the context's processor list is used.
"""
get_docs += select_docs

for attr in dir(Context):
    attr_val = getattr(Context, attr)
    if hasattr(attr_val, "__doc__"):
        doc = attr_val.__doc__
        if doc is not None and "{get_docs}" in doc:
            attr_val.__doc__ = doc.format(get_docs=get_docs)


@export
class OutsideException(Exception):
    pass
