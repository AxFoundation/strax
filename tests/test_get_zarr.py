
import strax
from strax.testutils import Records, Peaks, run_id
import tempfile
import numpy as np



def test_get_zarr():
    """Get a context for the tests below"""
    with tempfile.TemporaryDirectory() as temp_dir:
        context = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                       deep_scan=True),
                           register=[Records, Peaks],
                           use_per_run_defaults=True,
                           )
        records = context.get_array(run_id, 'records')
        peaks = context.get_array(run_id, 'peaks')
        zgrp = context.get_zarr(run_id, ('records', 'peaks'), storage='memory://')

    assert np.all(zgrp.records['time'] == records['time'])
    assert np.all(zgrp.peaks['time'] == peaks['time'])
