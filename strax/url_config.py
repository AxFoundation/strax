import json
import strax

import numbers
import inspect
from ast import literal_eval
from typing import Any, Dict, Optional, Mapping, Union

import pandas as pd
from urllib.parse import urlparse, parse_qs

try:
    # pydantic >= 2 ships the v1 API under pydantic.v1
    from pydantic.v1.validators import find_validators
    from pydantic.v1.config import get_config
except ImportError:
    from pydantic.validators import find_validators
    from pydantic.config import get_config

from strax.config import OMITTED
from strax.utils import filter_kwargs, CacheDict, total_size

export, __all__ = strax.exporter()

_CACHES: Dict[int, Any] = {}


@export
def clear_config_caches():
    for cache in _CACHES.values():
        cache.clear()


@export
def config_cache_size_mb():
    return total_size(_CACHES) // 1e6


def parse_val(val: str):
    """Attempt to parse a string value as a python literal, falls back to returning just the
    original string if cant be parsed."""
    try:
        val = literal_eval(val)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return val


@export
class URLConfig(strax.Config):
    """Dispatch on URL protocol.

    unrecognized protocol returns identity inspired by dasks Dispatch and fsspec fs protocols.

    """

    _LOOKUP: Dict[str, Any] = {}
    _PREPROCESSORS = ()

    SCHEME_SEP = "://"
    QUERY_SEP = "?"
    NAMESPACE_SEP = "."
    PLUGIN_ATTR_PREFIX = "plugin."

    def __init__(self, cache=0, **kwargs):
        """
        :param cache: number of values to keep in cache, if set to True will cache all values
        :param **kwargs: additional keyword arguments accepted by strax.Option
        """
        self.final_type = OMITTED
        super().__init__(**kwargs)
        # Ensure backwards compatibility with Option validation
        # type of the config value can be different from the fetched value.
        if self.type is not OMITTED:
            self.final_type = self.type
            self.type = OMITTED  # do not enforce type on the URL
        if cache:
            cache_len = 100 if cache is True else int(cache)
            cache = CacheDict(cache_len=cache_len)
            _CACHES[id(self)] = cache

    @property
    def cache(self):
        return _CACHES.get(id(self), {})

    @classmethod
    def register(cls, protocol, func=None):
        """Register dispatch of `func` on urls starting with protocol name `protocol`"""

        def wrapper(func):
            if isinstance(protocol, tuple):
                for t in protocol:
                    cls.register(t, func)
                return func

            if not isinstance(protocol, str):
                raise ValueError("Protocol name must be a string.")

            if protocol in cls._LOOKUP:
                raise ValueError(f"Protocol with name {protocol} already registered.")
            cls._LOOKUP[protocol] = func
            return func

        return wrapper(func) if func is not None else wrapper

    @classmethod
    def preprocessor(cls, func=None, precedence=0):
        """Register a new processor to modify the config values before they are used."""

        def wrapper(func):
            entry = (precedence, func)
            if entry in cls._PREPROCESSORS:
                raise ValueError(f"This processor is already registered.")
            cls._PREPROCESSORS += (entry,)
            return func

        return wrapper(func) if func is not None else wrapper

    @classmethod
    def eval(
        cls, protocol: str, arg: Optional[Union[str, tuple]] = None, kwargs: Optional[dict] = None
    ):
        """Evaluate a URL/AST by recusively dispatching protocols by name with argument arg and
        keyword arguments kwargs and return the value.

        If protocol does not exist, returnes arg
        :param protocol: name of the protocol or a URL
        :param arg: argument to pass to protocol, can be another (sub-protocol, arg, kwargs) tuple,
            in which case sub-protocol will be evaluated and passed to protocol
        :param kwargs: keyword arguments to be passed to the protocol
        :return: (Any) The return value of the protocol on these arguments

        """

        if protocol is not None and arg is None:
            protocol, arg, kwargs = cls.url_to_ast(protocol)

        if protocol is None:
            return arg

        if kwargs is None:
            kwargs = {}

        meth = cls._LOOKUP[protocol]

        if isinstance(arg, tuple):
            arg = cls.eval(*arg)

        # Just to be on the safe side
        kwargs = filter_kwargs(meth, kwargs)

        return meth(arg, **kwargs)

    @classmethod
    def split_url_kwargs(cls, url):
        """Split a url into path and kwargs."""
        path, _, _ = url.partition(cls.QUERY_SEP)
        kwargs = {}
        for k, v in parse_qs(urlparse(url).query).items():
            # values of query arguments are evaluated as lists
            # split logic depending on length
            n = len(v)
            if not n:
                kwargs[k] = None
            elif n == 1:
                kwargs[k] = parse_val(v[0])
            else:
                kwargs[k] = list(map(parse_val, v))
        return path, kwargs

    @classmethod
    def kwarg_from_url(cls, url: str, key: str):
        path, kwargs = cls.split_url_kwargs(url)
        return kwargs.get(key, None)

    @classmethod
    def format_url_kwargs(cls, url, **kwargs):
        """Add keyword arguments to a URL.

        Sorts all arguments by key for hash consistency

        """
        url, extra_kwargs = cls.split_url_kwargs(url)
        kwargs = dict(extra_kwargs, **kwargs)
        arg_list = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, list):
                # lists are passed as multiple arguments with the same key
                arg_list.extend([f"{k}={vi}" for vi in v])
            else:
                arg_list.append(f"{k}={v}")
        arg_str = "&".join(arg_list)
        arg_str = cls.QUERY_SEP + arg_str if arg_str else ""
        return url + arg_str

    @classmethod
    def lookup_value(cls, value, **namespace):
        """Optionally fetch an attribute from namespace if value is a string with cls.NAMESPACE_SEP
        in it, the string is split and the first part is used to lookup an object in namespace and
        the second part is used to lookup the value in the object.

        If the value is not a string or the target object is not in the namesapce, the value is
        returned as is.

        """

        if isinstance(value, list):
            return [cls.lookup_value(v, **namespace) for v in value]

        if isinstance(value, str) and cls.NAMESPACE_SEP in value:
            name, _, key = value.partition(cls.NAMESPACE_SEP)
            if name in namespace:
                obj = namespace[name]
                if isinstance(obj, Mapping):
                    value = obj.get(key, value)
                else:
                    value = getattr(obj, key, value)

        return value

    @classmethod
    def deref_ast(cls, protocol, arg, kwargs, **namespace):
        """Dereference an AST by looking up values in namespace."""
        if isinstance(arg, tuple):
            arg = cls.deref_ast(*arg, **namespace)
        else:
            arg = cls.lookup_value(arg, **namespace)
        kwargs = {k: cls.lookup_value(v, **namespace) for k, v in kwargs.items()}
        return protocol, arg, kwargs

    def validate(
        self,
        config,
        run_id=None,  # TODO: will soon be removed
        run_defaults=None,
        set_defaults=True,
    ):
        """This method is called by the context on plugin initialization at this stage, the run_id
        and context config are already known but the config values are not yet set on the plugin.

        Therefore its the perfect place to run any preprocessors on the config values to make any
        needed changes before the configs are hashed.

        """
        super().validate(config, run_id, run_defaults, set_defaults)

        cfg = config[self.name]

        sorted_preprocessors = reversed(sorted(self._PREPROCESSORS, key=lambda x: x[0]))

        full_kwargs = dict(
            name=self.name,
            run_id=run_id,
            run_defaults=run_defaults,
            set_defaults=set_defaults,
        )

        for _, preprocessor in sorted_preprocessors:
            kwargs = filter_kwargs(preprocessor, full_kwargs)
            new_cfg = preprocessor(cfg, **kwargs)
            cfg = new_cfg if new_cfg is not None else cfg

        config[self.name] = cfg

        if not isinstance(cfg, str) or self.SCHEME_SEP not in cfg:
            # if the value is not a url config it is validated against
            # its intended type (final_type)
            self.validate_type(cfg)

    def validate_type(self, value):
        """Validate the type of a value against its intended type."""
        msg = (
            f"Invalid type for option {self.name}. "
            f"Expected a {self.final_type} instance, got {type(value)}"
        )

        if self.final_type is not OMITTED:
            # Use pydantic to validate the type
            # its validation is more flexible than isinstance
            # it will coerce standard equivalent types
            cfg = get_config(dict(arbitrary_types_allowed=True))
            if isinstance(self.final_type, tuple):
                validators = [v for t in self.final_type for v in find_validators(t, config=cfg)]
            else:
                validators = find_validators(self.final_type, config=cfg)
            for validator in validators:
                try:
                    validator(value)
                    break
                except Exception:
                    pass
            else:
                raise TypeError(msg)

        return value

    def fetch(self, plugin):
        """Override the Config.fetch method this is called when the attribute is accessed from
        withing the Plugin instance."""
        # first fetch the user-set value

        # from the config dictionary
        url = super().fetch(plugin)

        if not isinstance(url, str):
            # if the value is not a string it is evaluated
            # as a literal config and returned as is.
            return self.validate_type(url)

        if self.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated
            # as string-literal config and returned as is
            return self.validate_type(url)

        # evaluate the url as AST
        protocol, arg, kwargs = self.url_to_ast(url)

        # allow run_id to be missing
        run_id = getattr(plugin, "run_id", "000000")

        # construct a deterministic hash key from AST
        key = strax.deterministic_hash((plugin.config, run_id, protocol, arg, kwargs))

        # fetch from cache if exists
        value = self.cache.get(key, None)

        # not in cache, lets fetch it
        if value is None:
            # resolve any referenced to plugin or config attributes
            protocol, arg, kwargs = self.deref_ast(
                protocol, arg, kwargs, config=plugin.config, plugin=plugin
            )

            value = self.eval(protocol, arg, kwargs)
            value = self.validate_type(value)
            self.cache[key] = value

        return value

    @classmethod
    def ast_to_url(
        cls,
        protocol: Union[str, tuple],
        arg: Optional[Union[str, tuple]] = None,
        kwargs: Optional[dict] = None,
    ):
        """Convert a protocol abstract syntax tree to a valid URL."""

        if isinstance(protocol, tuple):
            protocol, arg, kwargs = protocol

        if kwargs is None:
            kwargs = {}

        if protocol is None:
            return arg

        if isinstance(arg, (list, dict, numbers.Number)) and protocol != "json":
            arg = (
                "json",
                json.dumps(arg),
            )

        if isinstance(arg, tuple):
            arg = cls.ast_to_url(*arg)

        if not isinstance(arg, str):
            raise TypeError(f"Type {type(arg)} is not supported as an argument.")

        arg, extra_kwargs = cls.split_url_kwargs(arg)

        kwargs.update(extra_kwargs)

        url = f"{protocol}{cls.SCHEME_SEP}{arg}"

        url = cls.format_url_kwargs(url, **kwargs)

        return url

    @classmethod
    def url_to_ast(cls, url, **kwargs):
        """Convert a URL to a protocol abstract syntax tree."""
        if not isinstance(url, str):
            raise TypeError(f"URL must be a string, got {type(url)}")

        if cls.SCHEME_SEP not in url:
            # no protocol in the url so its evaluated
            # as string-literal config and returned as is
            return None, url, {}

        # separate the protocol name from the path
        protocol, _, path = url.partition(cls.SCHEME_SEP)

        # find the corresponding protocol method
        meth = cls._LOOKUP.get(protocol, None)
        if meth is None:
            # unrecognized protocol
            # evaluate as string-literal
            return None, url, {}

        arg, url_kwargs = cls.split_url_kwargs(path)
        kwargs.update(url_kwargs)

        if cls.SCHEME_SEP in arg:
            # url contains a nested protocol
            # first parsce sub-protocol
            arg = cls.url_to_ast(arg, **kwargs)

        # Filter unused kwargs for this method.
        # This is done also at the eval level but
        # probably better to be safe.
        kwargs = filter_kwargs(meth, kwargs)

        # Always sort kwargs for consistent ASTs
        kwargs = dict(sorted(kwargs.items()))

        return protocol, arg, kwargs

    @classmethod
    def are_equal(cls, first, second):
        """Return whether two URLs are equivalent (have equal ASTs)"""
        return cls.url_to_ast(first) == cls.url_to_ast(second)

    @classmethod
    def protocol_descr(cls):
        rows = []
        for k, v in cls._LOOKUP.items():
            descr = v.__doc__
            if descr is not None:
                descr = descr.split("\n")[0]

            row = {
                "name": f"{k}://",
                "description": descr,
                "signature": str(inspect.signature(v)),
                "location": v.__module__,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def print_protocols(cls):
        df = cls.protocol_descr()
        if len(df):
            print(df)
        else:
            print("No protocols registered.")

    @classmethod
    def preprocessor_descr(cls):
        rows = []
        for k, v in cls._PREPROCESSORS:
            descr = v.__doc__
            if descr is not None:
                descr = descr.split("\n")[0]
            row = {
                "precedence": k,
                "description": descr,
                "signature": str(inspect.signature(v)),
                "location": v.__module__,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def print_preprocessors(cls):
        df = cls.preprocessor_descr()
        if len(df):
            print(df)
        else:
            print("No Preprocessors registered.")

    @classmethod
    def print_summary(cls):
        print("=" * 30 + " Protocols " + "=" * 30)
        cls.print_protocols()
        print("=" * 30 + " Preprocessors " + "=" * 30)
        cls.print_preprocessors()

    @classmethod
    def evaluate_dry(cls, url: str, **kwargs):
        """Utility function to quickly test and evaluate URL configs, without the initialization of
        plugins (so no plugin attributes). plugin attributes can be passed as keyword arguments.

        example::

            from strax import URLConfig
            url_string='xedocs://electron_drift_velocity?run_id=027000&version=ONLINE'
            URLConfig.evaluate_dry(url_string)

            # or similarly
            url_string='xedocs://electron_drift_velocity?run_id=plugin.run_id&version=ONLINE'
            URLConfig.evaluate_dry(url_string, run_id='027000')

        Please note that this has to be done outside of the plugin, so any
        attributes of the plugin are not yet note to this dry evaluation
        of the url-string.

        :param url: URL to evaluate, see above for example.
        :keyword: any additional kwargs are passed to self.dispatch (see example)
        :return: evaluated value of the URL.

        """
        url = cls.format_url_kwargs(url, **kwargs)
        _, combined_kwargs = cls.split_url_kwargs(url)

        for k, v in combined_kwargs.items():
            if isinstance(v, str) and cls.PLUGIN_ATTR_PREFIX in v:
                raise ValueError(
                    f"The URL parameter {k} depends on the plugin. "
                    "You must specify the value for this parameter "
                    "for this URL to be evaluated correctly. "
                    f"Try passing {k} as a keyword argument. "
                    f"e.g.: `URLConfig.evaluate_dry({url}, {k}=SOME_VALUE)`."
                )

        return cls.eval(url)
