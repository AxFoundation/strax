.. strax documentation master file, created by
   sphinx-quickstart on Sat May  5 22:35:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
Strax
======

Github page: https://github.com/AxFoundation/strax

Strax is an analysis framework for pulse-only digitization data,
specialized for live data processing at speeds of 50-100 MB(raw) / core / sec.

Strax is initially developed for the XENONnT experiment, but the configuration
and specific algorithms for XENONnT will eventually be hosted separately.


.. toctree::
    :maxdepth: 1
    :caption: Setup and basics

    basics/setup
    basics/tutorial.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Advanced usage

    advanced/overview


.. toctree::
    :maxdepth: 1
    :caption: Developer documentation

    developer/chunking
    developer/pipeline
    developer/parallel

The above pages describe how strax's processing framework works under the hood, and explains some implementation choices. It's meant for people who want to do core development on strax; users or even plugin developers should not need it.

.. toctree::
    :maxdepth: 1
    :caption: API Reference

    reference/strax


* :ref:`genindex`
* :ref:`modindex`

