import builtins
import typing as ty

import strax
export, __all__ = strax.exporter()

# Placeholder value for omitted values.
# Use instead of None since None might be a proper value/default
OMITTED = '<OMITTED>'
__all__.append('OMITTED')


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
            plugin_class.takes_config.update(result)
        else:
            plugin_class.takes_config = result
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

    def get_default(self, run_id=None):
        """Return default value for the option"""
        if run_id is None:
            run_id = 0          # TODO: think if this makes sense
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

    def validate(self, config, run_id=None, set_defaults=True):
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
