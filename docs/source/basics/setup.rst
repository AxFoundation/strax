Setting up strax
================

To install the latest stable version (from pypi), run `pip install strax`.
Dependencies should install automatically:
numpy, pandas, numba, two compression libraries (blosc and zstd)
and a few miscellaneous pure-python packages. Strax requires python >= 3.6.

If you want to try out strax on XENON1T data, you're probably better off installing strax's XENON bindings at `<https://github.com/XENONnT/straxen>`_. Strax will be automatically installed along with straxen.

You might want to install some dependencies (such as numpy and numba) via conda rather than pip, but it's up to you.

You can also clone the repository, then setup a developer installation with `python setup.py develop`.

If you experience problems during installation, try installing
exactly the same version of the dependencies as used on the Travis build test server.
Clone the repository, then do `pip install -r requirements.txt`.
