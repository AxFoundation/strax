# strax
Streaming analysis for xenon experiments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1344424.svg)](https://doi.org/10.5281/zenodo.1344424)
[![Build Status](https://travis-ci.org/AxFoundation/strax.svg?branch=master)](https://travis-ci.org/AxFoundation/strax)
[![Readthedocs Badge](https://readthedocs.org/projects/strax/badge/?version=latest)](https://strax.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/AxFoundation/strax/badge.svg?branch=master)](https://coveralls.io/github/AxFoundation/strax?branch=master)
[![PyPI version shields.io](https://img.shields.io/pypi/v/strax.svg)](https://pypi.python.org/pypi/strax/)
[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/AxFoundation/strax?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Strax is an analysis framework for pulse-only digitization data, specialized for live data reduction at speeds of 50-100 MB(raw) / core / sec. For more information, please see the [strax documentation](https://strax.readthedocs.io).

Strax' primary aim is to support noble liquid TPC dark matter searches, such as XENONnT. The XENON-specific algorithms live in the separate package [straxen](https://github.com/XENONnT/straxen). If you want to try out strax, you probably want to start there. This package only contains the core framework and basic algorithms any TPCs would want to use.

