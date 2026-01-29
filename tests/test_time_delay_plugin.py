"""Tests for TimeDelayPlugin."""

import numpy as np
import strax
import pytest


def simple_interval_dtype():
    return [
        ("time", np.int64),
        ("length", np.int32),
        ("dt", np.int16),
        ("value", np.int32),
    ]


class ChunkedSource(strax.Plugin):
    """Source plugin that yields pre-defined chunks."""

    depends_on = tuple()
    provides = "source_data"
    dtype = simple_interval_dtype()
    rechunk_on_save = False
    chunks_data = []

    def is_ready(self, chunk_i):
        return chunk_i < len(self.chunks_data)

    def source_finished(self):
        return True

    def compute(self, chunk_i):
        start, end, data = self.chunks_data[chunk_i]
        return self.chunk(start=start, end=end, data=data)


class ConstantDelayPlugin(strax.TimeDelayPlugin):
    """Test plugin that adds a constant delay to all records."""

    depends_on = ("source_data",)
    provides = "delayed_data"
    dtype = simple_interval_dtype()
    data_kind = "delayed_data"
    delay = 0

    def get_max_delay(self):
        return self.delay

    def compute_with_delay(self, source_data):
        result = source_data.copy()
        result["time"] = result["time"] + self.delay
        return result


class VariableDelayPlugin(strax.TimeDelayPlugin):
    """Test plugin that adds variable delays based on a pattern."""

    depends_on = ("source_data",)
    provides = "variable_delayed_data"
    dtype = simple_interval_dtype()
    data_kind = "variable_delayed_data"
    max_delay = 100
    delay_pattern = [0]

    def get_max_delay(self):
        return self.max_delay

    def compute_with_delay(self, source_data):
        result = source_data.copy()
        delays = np.array(
            [self.delay_pattern[i % len(self.delay_pattern)] for i in range(len(result))]
        )
        result["time"] = result["time"] + delays
        return result


class MultiOutputDelayPlugin(strax.TimeDelayPlugin):
    """Test plugin with multiple outputs."""

    depends_on = ("source_data",)
    provides = ("delayed_output_a", "delayed_output_b")
    data_kind = {
        "delayed_output_a": "delayed_output_a",
        "delayed_output_b": "delayed_output_b",
    }
    delay_a = 0
    delay_b = 0
    max_delay = 100

    def infer_dtype(self):
        return {
            "delayed_output_a": simple_interval_dtype(),
            "delayed_output_b": simple_interval_dtype(),
        }

    def get_max_delay(self):
        return self.max_delay

    def compute_with_delay(self, source_data):
        result_a = source_data.copy()
        result_a["time"] = result_a["time"] + self.delay_a
        result_b = source_data.copy()
        result_b["time"] = result_b["time"] + self.delay_b
        return {"delayed_output_a": result_a, "delayed_output_b": result_b}


def make_test_data(times, length=1, dt=1, values=None):
    """Create test data array with given times."""
    n = len(times)
    data = np.zeros(n, dtype=simple_interval_dtype())
    data["time"] = times
    data["length"] = length
    data["dt"] = dt
    data["value"] = values if values is not None else np.arange(n)
    return data


def create_context_with_source(chunks_data):
    """Create a strax context with ChunkedSource configured."""

    class TestSource(ChunkedSource):
        pass

    TestSource.chunks_data = chunks_data
    st = strax.Context(storage=[])
    st.register(TestSource)
    return st


def test_constant_delay_across_chunks():
    """Test constant delay with buffering across chunk boundaries."""
    delay = 30

    data1 = make_test_data(np.array([10, 40]), values=np.array([0, 1]))
    data2 = make_test_data(np.array([60, 90]), values=np.array([2, 3]))

    chunks_data = [
        (0, 50, data1),
        (50, 100, data2),
    ]
    st = create_context_with_source(chunks_data)

    class TestDelayPlugin(ConstantDelayPlugin):
        delay = 30

    st.register(TestDelayPlugin)
    result = st.get_array(run_id="test", targets="delayed_data")

    expected_times = np.array([10, 40, 60, 90]) + delay
    np.testing.assert_array_equal(sorted(result["time"]), sorted(expected_times))
    assert len(result) == 4


def test_variable_delay_reorders_and_buffers():
    """Test variable delays with reordering and buffering."""
    data1 = make_test_data(np.array([0, 10, 20]), values=np.array([0, 1, 2]))
    data2 = make_test_data(np.array([50, 60, 70]), values=np.array([3, 4, 5]))

    chunks_data = [
        (0, 50, data1),
        (50, 100, data2),
    ]
    st = create_context_with_source(chunks_data)

    class TestVariableDelay(VariableDelayPlugin):
        max_delay = 100
        delay_pattern = [0, 80, 20]

    st.register(TestVariableDelay)
    result = st.get_array(run_id="test", targets="variable_delayed_data")

    assert len(result) == 6
    assert np.all(np.diff(result["time"]) >= 0), "Output must be sorted"


def test_empty_input_chunk():
    """Test handling of empty input chunks."""
    data1 = make_test_data(np.array([10, 20]), values=np.array([0, 1]))
    empty_data = make_test_data(np.array([], dtype=np.int64))
    data3 = make_test_data(np.array([110, 120]), values=np.array([2, 3]))

    chunks_data = [
        (0, 50, data1),
        (50, 100, empty_data),
        (100, 150, data3),
    ]
    st = create_context_with_source(chunks_data)

    class TestDelayPlugin(ConstantDelayPlugin):
        delay = 20

    st.register(TestDelayPlugin)
    result = st.get_array(run_id="test", targets="delayed_data")

    assert len(result) == 4


def test_multi_output_different_delays():
    """Test multi-output plugin with different delays per output."""
    data = make_test_data(np.array([10, 50]), values=np.array([0, 1]))
    chunks_data = [(0, 100, data)]
    st = create_context_with_source(chunks_data)

    class TestMultiOutput(MultiOutputDelayPlugin):
        delay_a = 20
        delay_b = 60
        max_delay = 60

    st.register(TestMultiOutput)

    result_a = st.get_array(run_id="test", targets="delayed_output_a")
    result_b = st.get_array(run_id="test", targets="delayed_output_b")

    np.testing.assert_array_equal(result_a["time"], [30, 70])
    np.testing.assert_array_equal(result_b["time"], [70, 110])


def test_chunk_continuity():
    """Test that output chunks maintain proper continuity."""
    data1 = make_test_data(np.array([10, 20]), values=np.array([0, 1]))
    data2 = make_test_data(np.array([60, 70]), values=np.array([2, 3]))
    data3 = make_test_data(np.array([110, 120]), values=np.array([4, 5]))

    chunks_data = [
        (0, 50, data1),
        (50, 100, data2),
        (100, 150, data3),
    ]
    st = create_context_with_source(chunks_data)

    class TestDelayPlugin(ConstantDelayPlugin):
        delay = 30

    st.register(TestDelayPlugin)
    chunks = list(st.get_iter(run_id="test", targets="delayed_data"))

    for i in range(1, len(chunks)):
        assert (
            chunks[i].start == chunks[i - 1].end
        ), f"Chunk {i} start ({chunks[i].start}) != chunk {i-1} end ({chunks[i-1].end})"


def test_straddling_data_across_boundary():
    """Test data that straddles chunk boundary (time < boundary < endtime)."""
    # After delay: time=95, endtime=105 (straddles boundary at 100)
    data1 = np.zeros(1, dtype=simple_interval_dtype())
    data1["time"] = 85
    data1["length"] = 10
    data1["dt"] = 1
    data1["value"] = 1

    data2 = np.zeros(1, dtype=simple_interval_dtype())
    data2["time"] = 120
    data2["length"] = 10
    data2["dt"] = 1
    data2["value"] = 2

    chunks_data = [
        (0, 100, data1),
        (100, 200, data2),
    ]
    st = create_context_with_source(chunks_data)

    class TestDelayPlugin(ConstantDelayPlugin):
        delay = 10

    st.register(TestDelayPlugin)
    result = st.get_array(run_id="test", targets="delayed_data")

    assert len(result) == 2
    np.testing.assert_array_equal(result["time"], [95, 130])
    np.testing.assert_array_equal(strax.endtime(result), [105, 140])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
