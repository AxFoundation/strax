import inspect
import numpy as np
import strax
from .plugin import Plugin, SaveWhen

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
        dtype = [(self.cut_name, np.bool_, self.cut_description)]
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
