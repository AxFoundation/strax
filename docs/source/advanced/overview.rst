Strax overview
==============

.. image:: architecture.svg


Features:

* Start from unordered streams of pulses (like pax's `trigger <https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:aalbers:trigger_upgrade>`_))

* Output to files or MongoDB

* Plugin system for extensibility

  * Each plugin produces a dataframe
  * Dependencies and configuration tracked explicitly
  * Limited "Event class" emulation for code that needs it

* Processing algorithms: hitfinding, sum waveform, clustering, classification, event building
