import builtins
import typing as ty
import numbers
from immutabledict import immutabledict
from numpy import isin
import inspect
from urllib.parse import urlparse, parse_qs
from ast import literal_eval
from functools import lru_cache
import warnings

import strax

export, __all__ = strax.exporter()

# Placeholder value for omitted values.
# Use instead of None since None might be a proper value/default
OMITTED = '<OMITTED>'
__all__ += 'OMITTED InvalidConfiguration'.split()


@export
class InvalidConfiguration(Exception):
    pass

@export
def takes_config(*options):
    """Decorator for plugin classes, to specify which options it takes.
    :param options: Option instances of options this plugin takes.
    """
    def wrapped(plugin_class):
        result = {}
        for opt in options:
            if not isinstance(opt, Option):
                raise RuntimeError("Specify config options by Option objects")
            opt.taken_by = plugin_class.__name__
            result[opt.name] = opt

        # For some reason the second condition is essential, I don't understand
        # yet why...
        if (hasattr(plugin_class, 'takes_config')
                and len(plugin_class.takes_config)):
            # Already have some options set, e.g. because of subclassing
            # where both child and parent have a takes_config decorator
            for opt in result.values():
                if opt.name in plugin_class.takes_config:
                    raise RuntimeError(
                        f"Attempt to specify option {opt.name} twice")
            plugin_class.takes_config = immutabledict({
                **plugin_class.takes_config, **result})
        else:
            plugin_class.takes_config = immutabledict(result)
        if isinstance(opt, strax.Config):
            setattr(plugin_class, opt.name, opt)
        return plugin_class

    return wrapped

def parse_val(val):
    try:
        val = literal_eval(val)
    except:
        pass
    return val


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
                 child_option: bool = False,
                 parent_option_name: str = None,
                 track: bool = True,
                 infer_dtype = OMITTED,
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
         "default_by_run" can only be usd in contexts where the context option
         "use_per_run_defaults" is set to True
        :param child_option: If true option is marked as a child_option. All
            options which are marked as a child overwrite the corresponding parent
            option. Removes also the corresponding parent option from the lineage.
        :param parent_option_name: Name of the parent option of child option.
            Required to find the key of the parent option so it can be overwritten
            by the value of the child option.
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

        # Options required for inherited child plugins:
        # Require both to be more explicit and reduce errors by the user
        self.child_option = child_option
        self.parent_option_name = parent_option_name
        if (self.child_option and not self.parent_option_name) \
                or (not self.child_option and self.parent_option_name):
            raise ValueError('You have to specify both, "child_option"=True and '
                             'the name of the parent option which should be '
                             'overwritten by the child. Options which are unique '
                             'to the child should not be marked as a child option.'
                             f'Please update {self.name} accordingly.')

        # if self.default_by_run is not OMITTED:
        #     warnings.warn(f"The {self.name} option uses default_by_run,"
        #                   f" which will soon stop working!",
        #                   DeprecationWarning)

        if sum([self.default is not OMITTED,
                self.default_factory is not OMITTED,
                self.default_by_run is not OMITTED]) > 1:
            raise RuntimeError(f"Tried to specify more than one default "
                               f"for option {self.name}.")
            
        if infer_dtype and type is OMITTED and default is not OMITTED:
            # ------------
            #FIXME: remove after long enough period to allow fixing problematic options.
            if infer_dtype is OMITTED:
                warnings.warn(f'You are setting a default value for config {name} but not \
                specifying a type. In the future the type will be inferred from \
                the default value which will result in an error if this config \
                is set to a different type.')
                return
            ## -----------
            for ntype in [numbers.Integral, numbers.Number]:
                # first check if its a number otherwise numpy numbers
                # will fail type checking when checked against int and float.
                # numbers.Integral, numbers.Number are safe to use
                # since numpy registers them as super-classes.
                # left as a loop since we may want to add other exceptions as
                # they are discovered.
                if isinstance(default, ntype):
                    self.type = ntype
                    break
            else:
                self.type = builtins.type(default)

    def get_default(self, run_id, run_defaults: dict = None):
        """Return default value for the option"""
        if run_defaults is not None and self.name in run_defaults:
            return run_defaults[self.name]
        if self.default is not OMITTED:
            return self.default
        if self.default_factory is not OMITTED:
            return self.default_factory()

        if self.default_by_run is not OMITTED:
            # TODO: This legacy code for handling default_per_run will soon
            # be removed!
            if run_id is None:
                run_id = 0  # TODO: think if this makes sense

            if isinstance(run_id, str):
                is_superrun = run_id.startswith('_')
                if not is_superrun:
                    run_id = int(run_id.replace('_', ''))
            else:
                is_superrun = False

            if callable(self.default_by_run):
                raise RuntimeError(
                    "Using functions to specify per-run defaults is no longer"
                    "supported: specify a (first_run, option) list, or "
                    "a URL of a file to process in the plugin")

            if is_superrun:
                return '<SUBRUN-DEPENDENT:%s>' % strax.deterministic_hash(
                    self.default_by_run)

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

    def validate(self, config,
                 run_id=None,   # TODO: will soon be removed
                 run_defaults=None, set_defaults=True):
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
            config[self.name] = self.get_default(run_id, run_defaults)


#Backward compatibility
@export
class Config(Option):
    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = ''
        super().__init__(**kwargs)

    def __set_name__(self, owner, name):
        self.name = name
        takes_config = {name: self}
        if (hasattr(owner, 'takes_config')
                and len(owner.takes_config)):
            # Already have some options set, e.g. because of subclassing
            # where both child and parent have a takes_config decorator
            
            if name in owner.takes_config:
                raise RuntimeError(
                    f"Attempt to specify option {name} twice")
            owner.takes_config = immutabledict({
                **owner.takes_config, **takes_config})
        else:
            owner.takes_config = immutabledict(takes_config)

    def __get__(self, obj, objtype=None):
        return self.fetch(obj)

    def __set__(self, obj, value):
        obj.config[self.name] = value

    def fetch(self, plugin):
        ''' This function is called when the attribute is being 
        accessed. Should be overridden by subclasses to customize behavior.
        '''
        if hasattr(plugin, 'config') and self.name in plugin.config:
            return plugin.config[self.name]
        raise AttributeError('Plugin has not been configured.')

@export
class LookupConfig(Config):
    mapping: ty.Mapping
    keys = ty.Iterable

    def __init__(self, mapping: ty.Mapping, keys=('name', 'value'), **kwargs):
        super().__init__(**kwargs)
        self.mapping = mapping
        if not isinstance(keys, ty.Iterable):
            keys = (keys,)
        self.keys = keys
        
    def fetch(self, plugin):
        key = []
        for k in self.keys:
            if k=='name':
                v = self.name
            elif k=='value':
                v = plugin.config[self.name]
            elif isinstance(k, str) and hasattr(plugin, k):
                v = getattr(plugin, k)
            else:
                v = k
            key.append(v)
        if len(key)==1:
            key = key[0]
        else:
            key = tuple(key)
        return self.mapping[key]


@export
class RemoteConfig(Config):
    storages: ty.Iterable
    name_key: str
    value_key: str
    
    def __init__(self, storages, name_key='name', value_key='value', **kwargs):
        super().__init__(**kwargs)
        self.storages = storages
        self.name_key = name_key
        self.value_key = value_key
        
    def fetch(self, plugin, **kwargs):
        kwargs[self.name_key] = self.name
        kwargs[self.value_key] = plugin.config[self.name]
        for store in self.storages:
            v = store.get_value(**kwargs)
            if v is not None:
                break
        else:
            raise KeyError(f'A value for the {self.name} config has not been \
                            found in any of its registered storages.')
        return v
    
@export
class CallableConfig(Config):
    func: ty.Callable

    def __init__(self, func: ty.Callable, args=(), kwargs: dict=None, **extra_kwargs):
        if not isinstance(func, ty.Callable):
            raise TypeError('func parameter must be of type Callable.')
        self.func = func
        self.args = args
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        super().__init__(**extra_kwargs)
    
    def fetch(self, plugin):
        args = []
        for arg in self.args:
            if isinstance(arg, str) and hasattr(plugin, arg):
                args.append(getattr(plugin, arg))
            else:
                args.append(arg)
            
        kwargs = {}
        for k,v in self.kwargs.items():
            if isinstance(v, str) and hasattr(plugin, v):
                kwargs[k] = getattr(plugin, v)
            else:
                kwargs[k] = v
        
        value = super().fetch(plugin)
        value = self.func(value, *args, **kwargs)
        return value


@export
class URLConfig(Config):
    """Dispatch on URL protocol.
    unrecognized protocol returns identity
    inspired by dasks Dispatch and fsspec fs protocols.
    """

    _lookup = {}
    _cache = {}

    def __init__(self, sep='://', attr_prefix='plugin.', cache=False, **kwargs):
        self.final_type = OMITTED
        super().__init__(**kwargs)
        # Ensure backwards compatibility with Option validation
        # type of the config value can be different from the fetched value.
        if self.type is not OMITTED:
            self.final_type = self.type
            self.type = OMITTED # do not enforce type on the URL
        self.sep = sep
        self.attr_prefix = attr_prefix
        if cache:
            self.dispatch = lru_cache()(self.dispatch)

    @classmethod
    def register(cls, protocol, func=None):
        """Register dispatch of `func` on urls
         starting with protocol name `protocol` """

        def wrapper(func):
            if isinstance(protocol, tuple):
                for t in protocol:
                    cls.register(t, func)
            else:
                cls._lookup[protocol] = func
            return func
        return wrapper(func) if func is not None else wrapper

    def dispatch(self, url, *args, **kwargs):
        """
        Call the corresponding method based on protocol in url.
        chained protocols will be called with the result of the
        previous protocol as input
        overrides are passed to any protocol whos signature can accept them.
        """
        if not isinstance(url, str):
            return url
        protocol, _, path =  url.partition(self.sep)

        meth = self._lookup.get(protocol, None)
        if meth is None:
            return url

        if self.sep in path:
            arg = self.dispatch(path, **kwargs)
        else:
            arg = path
        kwargs = self.filter_kwargs(meth, kwargs)
        return meth(arg, *args, **kwargs)
    
    @staticmethod
    def split_url_kwargs(url):
        arg, _, _ = url.partition('?')
        kwargs = {}
        for k,v in parse_qs(urlparse(url).query).items():
            n = len(v)
            if not n:
                kwargs[k] = None
            elif n==1:
                kwargs[k] = parse_val(v[0])
            else:
                kwargs[k] = map(parse_val, v)
        return arg, kwargs
    
    @staticmethod
    def filter_kwargs(func, kwargs):
        params = inspect.signature(func).parameters
        if any([str(p).startswith('**') for p in params.values()]):
            return kwargs
        return {k:v for k,v in kwargs.items() if k in params}

    def fetch(self, plugin):
        url = super().fetch(plugin)
        if not isinstance(url, str):
            return url
        if self.sep not in url:
            return url 
        url, url_kwargs = self.split_url_kwargs(url)
        kwargs = {}
        for k,v in url_kwargs.items():
            if isinstance(v, str) and v.startswith(self.attr_prefix):
                kwargs[k] = getattr(plugin, v[len(self.attr_prefix):], v)
            else:
                kwargs[k] = v
        
        return self.dispatch(url, **kwargs)


@export
def combine_configs(old_config, new_config=None, mode='update'):
    if new_config is None:
        new_config = dict()

    if mode == 'update':
        c = old_config.copy()
        c.update(new_config)
        return c
    if mode == 'setdefault':
        return combine_configs(new_config, old_config, mode='update')
    if mode == 'replace':
        return new_config

    raise RuntimeError("Expected update, setdefault or replace as config "
                       "setting mode")
