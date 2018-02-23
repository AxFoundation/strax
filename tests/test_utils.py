import numpy as np
import strax


def test_growing_result():
    @strax.growing_result(np.int, chunk_size=2)
    def bla(buffer):
        offset = 0

        for i in range(5):
            buffer[offset] = i

            offset += 1
            if offset == len(buffer):
                yield offset
                offset = 0
        yield offset

    result = np.array([0, 1, 2, 3, 4], dtype=np.int)
    np.testing.assert_equal(bla(), result)
    np.testing.assert_equal(bla(chunk_size=1), result)
    np.testing.assert_equal(bla(chunk_size=7), result)
    np.testing.assert_equal(bla(dtype=np.float), result.astype(np.float))
