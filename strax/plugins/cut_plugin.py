import inspect
import numpy as np
import strax
from .plugin import Plugin, SaveWhen
from .merge_only_plugin import MergeOnlyPlugin

export, __all__ = strax.exporter()


@export
class CutPlugin(Plugin):
    """Generate a plugin that provides a boolean for a given cut specified by 'cut_by'."""

    save_when = SaveWhen.TARGET

    def __init__(self):
        super().__init__()

        compute_pars = list(inspect.signature(self.cut_by).parameters.keys())
        if "chunk_i" in compute_pars:
            self.compute_takes_chunk_i = True
            del compute_pars[compute_pars.index("chunk_i")]
        if "start" in compute_pars:
            if "end" not in compute_pars:
                raise ValueError(f"Compute of {self} takes start, so it should also take end.")
            self.compute_takes_start_end = True
            del compute_pars[compute_pars.index("start")]
            del compute_pars[compute_pars.index("end")]
        self.compute_pars = compute_pars

        _name = strax.camel_to_snake(self.__class__.__name__)
        if not hasattr(self, "provides"):
            self.provides = _name
        if not hasattr(self, "cut_name"):
            self.cut_name = _name
        if not hasattr(self, "cut_description"):
            _description = _name
            if "cut_" not in _description:
                _description = "Cut by " + _description
            else:
                _description = " ".join(_description.split("_"))
            self.cut_description = _description

    def infer_dtype(self):
        dtype = [(self.cut_name, bool, self.cut_description)]
        # Alternatively one could use time_dt_fields for low level plugins.
        dtype = strax.time_fields + dtype
        return dtype

    def compute(self, **kwargs):
        if hasattr(self, "cut_by"):
            cut_by = self.cut_by
        else:
            raise NotImplementedError(f"{self.cut_name} does not have attribute 'cut_by'")

        # Take shape of the first data_type like in strax.plugin
        buff = list(kwargs.values())[0]

        # Generate result buffer
        r = np.zeros(len(buff), self.dtype)
        r["time"] = buff["time"]
        r["endtime"] = strax.endtime(buff)
        r[self.cut_name] = cut_by(**kwargs)
        return r

    def cut_by(self, **kwargs):
        # This should be provided by the user making a CutPlugin
        raise NotImplementedError()


@export
class CutList(MergeOnlyPlugin):
    """Base class that merges all existing cuts into a single array which can be loaded by the
    analysts."""

    __version__ = "0.0.0"

    save_when = SaveWhen.TARGET
    cuts = ()
    # need to declare depends_on here to satisfy strax
    # https://github.com/AxFoundation/strax/blob/df18c9cef38ea1cee9737d56b1bea078ebb246a9/strax/plugin.py#L99
    depends_on = ()
    _depends_on = ()

    def infer_dtype(self):
        dtype = super().infer_dtype()
        dtype += [
            (
                (
                    f"Boolean AND of all cuts in {self.accumulated_cuts_string}",
                    self.accumulated_cuts_string,
                ),
                bool,
            )
        ]
        return dtype

    def compute(self, **kwargs):
        cuts = super().compute(**kwargs)
        cuts_joint = np.zeros(len(cuts), self.dtype)
        strax.copy_to_buffer(
            cuts, cuts_joint, f"_copy_cuts_{strax.deterministic_hash(self.depends_on)}"
        )
        cuts_joint[self.accumulated_cuts_string] = get_accumulated_bool(cuts)
        return cuts_joint

    @property  # type: ignore
    def depends_on(self):  # noqa
        if not len(self._depends_on):
            deps = []
            for c in self.cuts:
                deps.extend(strax.to_str_tuple(c.provides))
            self._depends_on = tuple(deps)
        return self._depends_on

    @depends_on.setter
    def depends_on(self, str_or_tuple):
        self._depends_on = strax.to_str_tuple(str_or_tuple)


@export
def get_accumulated_bool(array):
    """Computes accumulated boolean over all cuts.

    :param array: Array containing merged cuts.

    """
    fields = array.dtype.names
    fields = np.array([f for f in fields if f not in ("time", "endtime")])

    res = np.ones(len(array), bool)
    for field in fields:
        res &= array[field]
    return res
