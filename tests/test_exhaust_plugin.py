from typing import Tuple
import numpy as np
import strax
from strax import Plugin, ExhaustPlugin


@strax.takes_config(
    strax.Option(name="n_chunks", default=10),
    strax.Option(name="n_items", default=10),
)
class ToExhaust(Plugin):
    depends_on: Tuple = tuple()
    provides: str = "to_exhaust"

    dtype = strax.time_fields

    source_done = False

    def compute(self, chunk_i):
        data = np.empty(self.config["n_items"], dtype=self.dtype)
        data["time"] = np.arange(self.config["n_items"]) + chunk_i * self.config["n_items"]
        data["endtime"] = data["time"]

        if chunk_i == self.config["n_chunks"] - 1:
            self.source_done = True

        return self.chunk(
            data=data,
            start=int(data[0]["time"]),
            end=int(strax.endtime(data[-1])) + 1,  # to make sure that data is continuous
        )

    def source_finished(self):
        return self.source_done

    def is_ready(self, chunk_i):
        if "ready" not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
        return self.ready


@strax.takes_config(
    strax.Option(name="n_chunks", default=10),
    strax.Option(name="n_items", default=10),
)
class Exhausted(ExhaustPlugin):
    depends_on: str = "to_exhaust"
    provides: str = "exhausted"

    dtype = strax.time_fields

    def compute(self, to_exhaust):
        return to_exhaust

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        flag = self.input_buffer[d] is None  # only check if we have not read anything yet
        super()._fetch_chunk(d, iters, check_end_not_before=check_end_not_before)
        if flag and (len(self.input_buffer[d]) != self.config["n_chunks"] * self.config["n_items"]):
            raise RuntimeError("Exhausted plugin did not read all chunks!")
        return False


def test_exhaust_plugin():
    """Test the ExhaustPlugin, about whether it can really exhaust the data or not."""
    st = strax.Context(storage=[])
    st.register((ToExhaust, Exhausted))
    st.storage = [
        strax.DataDirectory(
            "./strax_data",
            provide_run_metadata=True,
        )
    ]
    run_id = "000000"
    st.make(run_id, "to_exhaust")
    st.get_array(run_id, "exhausted")
