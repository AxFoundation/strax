# strax
Stream analysis for XENON (experimental)

Strax is a processor for pulse-only digitization data, 
specialized for live data reduction at speeds of 50-100 MB(raw) / core / sec. 

While more limited in scope than the XENON1T processor [pax](https://github.com/XENON1T/pax), it runs much faster.
This is due to using [numpy](https://docs.scipy.org/doc/numpy/) [structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html) internally,
which are supported by the amazing just-in-time compiler [numba](http://numba.pydata.org/).

Like pax's [trigger](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:aalbers:trigger_upgrade), strax works on unordered streams of pulses rather than pre-built event ranges.
Unlike pax's trigger, it processes the actual pulse data rather than looking only at pulse start times.
