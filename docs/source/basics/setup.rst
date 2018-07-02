Setting up strax
================

Installation
------------
To install the latest stable version (from pypi), run `pip install strax`.
Dependencies should install automatically:
numpy, pandas, numba, two compression libraries (blosc and zstd)
and a few miscellaneous pure-python packages.

If you also want to run tests with XENON1T data, run ``pip install strax[xenon]``.
This will also install keras and tensorflow, which are needed for the XENON1T neural net.
You might want to install these via conda rather than pip, but it's up to you.

You can also clone the repository, then setup a developer installation with `python setup.py develop`.

If you experience problems during installation, try installing
exactly the same version of the dependencies as used on the Travis build test server.
Clone the repository, then do `pip install -r requirements.txt`.

Downloading test data
----------------------
The provided demonstration notebooks require test data that is not included in the repository. Eventually we will provide simulated data for this purpose.

For now, only XENON collaboration members can find test data at:
   * `Processed only <https://xe1t-wiki.lngs.infn.it/lib/exe/fetch.php?media=xenon:xenon1t:aalbers:processed.zip>`_ (for strax demo notebook)
   * Raw (for fake_daq.py and eb.py) at midway: `/scratch/midway2/aalbers/test_input_data.zip`

To use these, unzip them in the same directory as the notebooks. The 'processed' zipfile will just make one directory with a single zipfile inside.
