Options and defaults
=====================

You can specify options using the ``strax.takes_config`` decorator and the ``strax.Option`` objects. See any plugin source code for example.

There is a single configuration dictionary in a strax context, shared by all plugins. Be judicious in how you name your options to avoid clashes. "Threshold" is probably a bad name, "peak_min_channels" is better.

If an option is not set in the context's configuration, strax will use its default value if the option defines one. If the plugin specifies no default, you will get a RuntimeError when attempting to start processing.

Even when a default is used, the value used will be stored in the metadata of the produced data. Thus, if you only change a default value for a plugin's option, you do NOT have to increment the plugin version to ensure data is rebuilt when needed.

You can specify defaults in several ways:

- ``default``: Use the given value as default.
- ``default_factory``: Call the given function (with no arguments) to produce a default. Use for mutable values such as lists.
- ``default_per_run``: Specify a list of 2-tuples: ``(start_run, default)``. Here start_run is a numerized run name (e.g ``170118_1327``; note the underscore is valid in integers since python 3.6) and ``default`` the option that applies from that run onwards.
- The ``strax_defaults`` dictionary in the run metadata. This overrides any defaults specified in the plugin code, but take care -- if you change a value here, there will be no record anywhere of what value was used previously, so you cannot reproduce your results anymore!


Example
________

.. code-block:: python

    @strax.takes_config(
        strax.Option('config_name', type=int, default=1)
    )
    class DummyPlugin(strax.Plugin):
        depends_on = ('records', )
        provides = ('dummy_data')
        ...

        def compute(self, records):
            value = self.config_name
            # or
            value = self.config['config_name']
            ...


Descriptor Options
------------------
An alternative way to define plugin configuration is with the ``Config`` class as follows:

.. code-block:: python

    class DummyPlugin(strax.Plugin):
        depends_on = ('records', )

        config_name = Config(type=int, default=1)

        def compute(self, records):
            # configs should be accessed as attributes for runtime evaluation
            value = self.config_name*2


Some projects require more flexible plugin configuration that is evaluated at runtime.
For these cases its recommended to subclass the ``Config`` class and overwrite the ``fetch(self, plugin)`` method
to compute the value from the current plugin state at runtime when the attribute is accessed.

A few tips when implementing such workflows:
  - You should limit yourself to a single syntax for your plugin configuration. Mixing multiple approaches in a single project can increase the complexity and mental burdon on analysts who will need to remember multiple configuratoin syntaxes and which one is used in each case.
  - Remember that whatever syntax is used, strax assumes the same set of user configs will always create the same data. When defining complex lookups for the plugin configuration at runtime it is up to you to keep this implicit promise.
  - When defining time-consuming lookups, it is recommended to implement a caching mechanism. Configuration value may be accessed many times during processing and expensive runtime computation of these values can reduce performance significantly.


Reference implementations
_________________________

Lookup by key

.. code-block:: python

    import strax
    import typing as ty


    class LookupConfig(Config):
        mapping: ty.Mapping
        keys = ty.Iterable

        def __init__(self, mapping: ty.Mapping, keys=('name', 'value'), **kwargs):
            super().__init__(**kwargs)
            self.mapping = mapping
            keys = strax.to_str_tuple(keys)
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

Find config from a list of values stores.

.. code-block:: python

    import strax
    import typing as ty


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


Fetch config value from a callable

.. code-block:: python

    import strax
    import typing as ty


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

URL style configuration (used in `straxen <https://github.com/XENONnT/straxen>`_ )

.. code-block:: python

    import strax
    from numpy import isin
    import inspect
    from urllib.parse import urlparse, parse_qs
    from ast import literal_eval
    from functools import lru_cache

    def parse_val(val):
        try:
            val = literal_eval(val)
        except:
            pass
        return val

    class URLConfig(strax.Config):
        """Dispatch on URL protocol.
        unrecognized protocol returns identity
        inspired by dasks Dispatch and fsspec fs protocols.
        """

        _LOOKUP = {}
        SCHEME_SEP = '://'
        QUERY_SEP = '?'
        PLUGIN_ATTR_PREFIX = 'plugin.'

        def __init__(self, cache=False, **kwargs):
            self.final_type = OMITTED
            super().__init__(**kwargs)
            # Ensure backwards compatibility with Option validation
            # type of the config value can be different from the fetched value.
            if self.type is not OMITTED:
                self.final_type = self.type
                self.type = OMITTED # do not enforce type on the URL
            if cache:
                maxsize = cache if isinstance(cache, int) else None
                self.dispatch = lru_cache(maxsize)(self.dispatch)

        @classmethod
        def register(cls, protocol, func=None):
            """Register dispatch of `func` on urls
            starting with protocol name `protocol` """

            def wrapper(func):
                if isinstance(protocol, tuple):
                    for t in protocol:
                        cls.register(t, func)
                    return func

                if not isinstance(protocol, str):
                    raise ValueError('Protocol name must be a string.')

                if protocol in cls._LOOKUP:
                    raise ValueError(f'Protocol with name {protocol} already registered.')
                cls._LOOKUP[protocol] = func
                return func
            return wrapper(func) if func is not None else wrapper

        def dispatch(self, url, *args, **kwargs):
            """
            Call the corresponding method based on protocol in url.
            chained protocols will be called with the result of the
            previous protocol as input
            overrides are passed to any protocol whos signature can accept them.
            """

            # seperate the protocol name from the path
            protocol, _, path =  url.partition(self.SCHEME_SEP)

            # find the corresponding protocol method
            meth = self._LOOKUP.get(protocol, None)
            if meth is None:
                # unrecongnized protocol
                # evaluate as string-literal
                return url

            if self.SCHEME_SEP in path:
                # url contains a nested protocol
                # first call sub-protocol
                arg = self.dispatch(path, **kwargs)
            else:
                # we are at the end of the chain
                # method should be called with path as argument
                arg = path

            # filter kwargs to pass only the kwargs
            #  accepted by the method.
            kwargs = self.filter_kwargs(meth, kwargs)

            return meth(arg, *args, **kwargs)

        def split_url_kwargs(self, url):
            """split a url into path and kwargs
            """
            path, _, _ = url.rpartition(self.QUERY_SEP)
            kwargs = {}
            for k,v in parse_qs(urlparse(url).query).items():
                # values of query arguments are evaluated as lists
                # split logic depending on length
                n = len(v)
                if not n:
                    kwargs[k] = None
                elif n==1:
                    kwargs[k] = parse_val(v[0])
                else:
                    kwargs[k] = map(parse_val, v)
            return path, kwargs

        @staticmethod
        def filter_kwargs(func, kwargs):
            """Filter out keyword arguments that
                are not in the call signature of func
                and return filtered kwargs dictionary
            """
            params = inspect.signature(func).parameters
            if any([str(p).startswith('**') for p in params.values()]):
                # if func accepts wildcard kwargs, return all
                return kwargs
            return {k:v for k,v in kwargs.items() if k in params}

        def fetch(self, plugin):
            # first fetch the user-set value
            # from the config dictionary
            url = super().fetch(plugin)

            if not isinstance(url, str):
                # if the value is not a string it is evaluated
                # as a literal config and returned as is.
                return url

            if self.SCHEME_SEP not in url:
                # no protocol in the url so its evaluated
                # as string-literal config and returned as is
                return url

            # sperate out the query part of the URL which
            # will become the method kwargs
            url, url_kwargs = self.split_url_kwargs(url)

            kwargs = {}
            for k,v in url_kwargs.items():
                if isinstance(v, str) and v.startswith(self.PLUGIN_ATTR_PREFIX):
                    # kwarg is referring to a plugin attribute, lets fetch it
                    kwargs[k] = getattr(plugin, v[len(self.PLUGIN_ATTR_PREFIX):], v)
                else:
                    # kwarg is a literal, add its value to the kwargs dict
                    kwargs[k] = v

            return self.dispatch(url, **kwargs)

