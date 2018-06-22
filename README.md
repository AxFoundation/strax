# strax
Streaming analysis for Xenon experiments

[![Build Status](https://travis-ci.org/AxFoundation/strax.svg?branch=master)](https://travis-ci.org/AxFoundation/strax)
[![Coverage Status](https://coveralls.io/repos/github/AxFoundation/strax/badge.svg?branch=master)](https://coveralls.io/github/AxFoundation/strax?branch=master)
[![PyPI version shields.io](https://img.shields.io/pypi/v/strax.svg)](https://pypi.python.org/pypi/strax/)
[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/AxFoundation/strax?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cc159474f2764d43b445d562a24ca245)](https://www.codacy.com/app/tunnell/strax?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=AxFoundation/strax&amp;utm_campaign=Badge_Grade)

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
  * [Tutorial notebook](https://www.github.com/AxFoundation/strax/blob/master/notebooks/Strax%20demo.ipynb)
  * [Introductory talk](https://docs.google.com/presentation/d/1qZmbAKJmzn7iTbBbkzhTvHmiBqdbYyxhgheRRrDhTeY) (aimed at XENON1T analysis/DAQ experts)
  * Function reference (TODO readthedocs)


