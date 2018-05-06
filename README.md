# strax
Streaming analysis for Xenon experiments

[![Build Status](https://travis-ci.org/AxFoundation/strax.svg?branch=master)](https://travis-ci.org/JelleAalbers/strax)
[![Coverage Status](https://coveralls.io/repos/github/AxFoundation/strax/badge.svg?branch=master)](https://coveralls.io/github/JelleAalbers/strax?branch=master)
[![PyPI version shields.io](https://img.shields.io/pypi/v/strax.svg)](https://pypi.python.org/pypi/strax/)
[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/AxFoundation/strax?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Strax is an analysis framework for pulse-only digitization data, 
specialized for live data reduction at speeds of 50-100 MB(raw) / core / sec. 

For comparison, this is more than 100x faster than the XENON1T processor [pax](http://github.com/XENON1T/pax),
and does not require a preprocessing stage ('eventbuilder').
It achieves this due to using [numpy](https://docs.scipy.org/doc/numpy/) [structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html) internally,
which are supported by the amazing just-in-time compiler [numba](http://numba.pydata.org/).

Features:
  * Start from unordered streams of pulses (like pax's [trigger](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:aalbers:trigger_upgrade))
  * Output to files or MongoDB
  * Plugin system for extensibility
    * Each plugin produces a dataframe
    * Dependencies and configuration tracked explicitly
    * Limited "Event class" emulation for code that needs it
  * Processing algorithms: hitfinding, sum waveform, clustering, classification, event building
 
Strax is initially developed for the XENONnT experiment. However, the configuration
and specific algorithms for XENONnT will ultimately be hosted into a separate repository.

### Documentation

Documentation is under construction. For the moment, you might find these useful:
  * [Tutorial notebook](https://www.github.com/JelleAalbers/strax/blob/master/notebooks/Strax%20demo.ipynb)
  * [Introductory talk](https://docs.google.com/presentation/d/1qZmbAKJmzn7iTbBbkzhTvHmiBqdbYyxhgheRRrDhTeY) (aimed at XENON1T analysis/DAQ experts)
  * Function reference (TODO readthedocs)


### Installation
To install the latest stable version (from pypi), run `pip install strax`. 
Dependencies should install automatically: 
numpy, pandas, numba, two compression libraries (blosc and zstd
and a few miscellaneous pure-python packages.

You can also clone the repository, then setup a developer installation with `python setup.py develop`.

If you experience problems during installation, try installing 
exactly the same version of the dependencies as used on he Travis build test server. 
Clone the repository, then do `pip install -r requirements.txt`.

#### Test data

The provided demonstration notebooks require test data that is not included in the repository.
Eventually we will provide simulated data for this purpose.
For now, XENON collaboration members can find test data at:
   * [Processed only](https://xe1t-wiki.lngs.infn.it/lib/exe/fetch.php?media=xenon:xenon1t:aalbers:processed.zip) (for strax demo notebook) 
   * Raw (for fake_daq.py and eb.py) at midway: `/scratch/midway2/aalbers/test_input_data.zip`

To use these, unzip them in the same directory as the notebooks. The 'processed' zipfile will just make one directory with a single zipfile inside.


