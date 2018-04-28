"""Configuration validation

TODO: Separate config sections?
"""
import typing
import warnings
import builtins

import strax
export, __all__ = strax.exporter()


@export
class InvalidConfiguration(Exception):
    pass


# Todo:  Does it really have to be a decorator just for nice error message?
@export
def takes_config(*options):
    def wrapped(plugin_class):
        result = []
        for opt in options:
            if isinstance(opt, str):
                opt = Option(opt)
            elif not isinstance(opt, Option):
                raise RuntimeError("Specify config options by str or Option")
            opt.taken_by = plugin_class.__name__
            result.append(opt)

        plugin_class.takes_config = {opt.name: opt for opt in result}
        return plugin_class

    return wrapped


# Use instead of None since None might be a proper value/default
OMITTED = 'Argument was not given'


@export
class Option:
    taken_by = "UNKNOWN???"

    def __init__(self,
                 name: str,
                 type: type = OMITTED,
                 default: typing.Any = OMITTED,
                 default_factory: typing.Callable = OMITTED,
                 help: str = ''):
        self.name = name
        self.type = type
        type = builtins.type
        self.default = default
        self.default_factory = default_factory
        self.help = help

        if (self.default is not OMITTED
                and self.default_factory is not OMITTED):
            raise RuntimeError(f"Tried to specify both default and "
                               f"default_factory for option {self.name}")

        if type is OMITTED and default is not OMITTED:
            self.type = type(default)

    def validate(self, config):
        """Checks if the option is in config and sets default if needed.
        """
        if self.name in config:
            value = config[self.name]
            if (self.type is not OMITTED
                    and not isinstance(value, self.type)):
                raise InvalidConfiguration(
                    f"Invalid type for option {self.name}. "
                    f"Excepted a {self.type}, got a {type(value)}")
        else:
            if self.default is OMITTED:
                if self.default_factory is OMITTED:
                    raise InvalidConfiguration(f"Missing option {self.name} "
                                               f"required by {self.taken_by}")
                config[self.name] = self.default_factory()
            else:
                config[self.name] = self.default


@export
def validate_config(config, plugins):
    for p in plugins:
        for opt in p.takes_config.values():
            opt.validate(config)

    all_opts = set().union(*[p.takes_config.keys() for p in plugins])
    for k in config:
        if k not in all_opts:
            warnings.warn(f"Option {k} not taken by any registered plugin")


# TODO: Hm, should't this be a plugin method?
@export
def set_plugin_config(config, plugin):
    plugin.config = {k: v for k, v in config.items()
                     if k in plugin.takes_config}
