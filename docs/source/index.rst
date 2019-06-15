======
Strax
======

Github page: https://github.com/AxFoundation/strax

Strax is an analysis framework for pulse-only digitization data,
specialized for live data processing at speeds of 50-100 MB(raw) / core / sec.

For comparison, this is more than 100x faster than the XENON1T processor `pax <http://github.com/XENON1T/pax>`_,
and does not require a preprocessing stage ('eventbuilder').
It achieves this due to using `numpy <https://docs.scipy.org/doc/numpy/>`_ `structured arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ internally,
which are supported by the amazing just-in-time compiler `numba <http://numba.pydata.org/>`_.

Strax is primarily developed for the XENONnT experiment, although the configuration and specific algorithms for XENONnT are hosted at `<https://github.com/XENONnT/straxen>`_. You can find its documentation `here <https://straxen.readthedocs.io>`_.

You might also find these presentations useful:

* `Talk on strax at the first XENONnT software telecon (May 2018) <https://docs.google.com/presentation/d/1khf-RNp6K-Q3TW1nQr5xUdrCUPGTJ8lDlDxnAh3s__U>`_
* `Talk on strax for DAQ experts (May 2018) <https://docs.google.com/presentation/d/1qZmbAKJmzn7iTbBbkzhTvHmiBqdbYyxhgheRRrDhTeY>`_


.. toctree::
    :maxdepth: 1
    :caption: Setup and basics

    basics/setup

.. toctree::
    :maxdepth: 1
    :caption: Advanced usage

    advanced/overview
    advanced/plugin_dev

.. toctree::
    :maxdepth: 1
    :caption: Developer documentation

    developer/chunking
    developer/pipeline
    developer/parallel
    developer/overlaps
    developer/storage
    developer/contributing
    developer/release

The above pages describe how strax's processing framework works under the hood, and explains some implementation choices. It's meant for people who want to do core development on strax; users or even plugin developers should not need it.

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    reference/strax


* :ref:`genindex`
* :ref:`modindex`

