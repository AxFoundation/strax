0.8.8 / 2020-02-11
------------------
- Fixes for time range selection (#231)
- Mailbox timeout and max_messages accessible as context options
- Fix output inference for ParallelSourcePlugin (straxen #46)
- Sane default for max_workers in multi-run loading
- run_id field is now a string when loading multiple runs

0.8.7 / 2020-02-07
------------------
- Small bugfixes:
  - Fixes for multi-output plugins
  - Use frozendict for Plugin.takes_config 

0.8.6 / 2020-01-17
-------------------
- Peak merging code (from straxen)
- time_range selection for user-defined plugins that never save
- Add hit height, store fpart-baseline corrected hit area
- Preserve cached input on empty chunks in OverlapWindowPlugin

0.8.5 / 2020-01-16
------------------
- Natural breaks splitting (#225)
- Add ptype, max_gap and max_goodness_of_split to peaks dtype
- Fixes for multiprocessing
- Fixes for time selection
- Fix for final data in same-length merging

0.8.4 / 2019-12-24
------------------
- Export downsampling function (#224)
- Fix show_config
- Fix pulse_processing for empty chunks

0.8.3 / 2019-12-23
------------------
- Fix for skipping data near end of run during multi-kind merge
- Add tight coincidence field to peaks dtype
- Pulse filtering optimization
- `max_messages` configurable per plugin, defaults to 4

0.8.2 / 2019-12-19
------------------
- Specify defaults via run doc (#223)
- Fix hypothesis test deadline issue during build (5bf2ad7)
- Fix: use selection_str also when selecting time range (87faeab)

0.8.1 / 2019-11-13
------------------
- Protect OverlapWindowPlugin against empty chunks (#212)
- Make test helpers accessible, test with numba on (#219)

0.8.0 / 2019-09-16
------------------
- Superruns (#207)
- Pulse processing fixes (#207)
- LZ4 compression (#207)
- Fixes for edge cases (#201)

0.7.5 / 2019-07-06
------------------
- Time selection refactor and context extensibility (#195)

0.7.4 / 2019-06-26
-------------------
- Fix availability checks (#194)
- Allow selection of runs by name (#192)
- Fix some context methods for multi-output plugins

0.7.3 / 2019-06-17
-------------------
- Multiple outputs per plugin (#190)
- Minor fixes and additions (#188, #182, #175, #185)

0.7.2 / 2019-06-06
------------------
- Area per channel in PE (#187)
- Update pinned dependencies, notably numba to 0.44.0 (#186)
- Fixes to empty chunk handling and chunk_arrays

0.7.1 / 2019-05-11
------------------
- Sum waveform now operates on all channels (#158)
- MongoDB output (#159)
- Better exception handling in saver (#160)
- Force plugins to produce correct dtype (#161)

0.7.0 / 2019-05-04
------------------
- Pulse processing upgrades (filtering etc) (#154)
- Run selection and run-level metadata handling (#155)
- Stabilize and shorten lineage hash (#152)
- Shared memory transfers, parallel save/load (#150)
- Ensure unique filenames (#143)
- Many processing fixes (#134, #129)

0.6.1 / 2019-01-20
-------------------
- Many bugfixes from DAQ test (#118)
- Fix dtype merging bug, add saturation info (#120)
- Fixes to sum waveform (cd0cd2f)

0.6.0 / 2018-10-09
------------------
- strax / straxen split (#107)
- Support incomplete data loading (#99)
- Fix for loading data made by ParallelSourcePlugin (#104)
- Runs DB frontend (#100) (moved to straxen)
- Fix MANIFEST.in

0.5.0 / 2018-09-02
------------------
- Directory name delimiter changed from `_` to `-` (#76)
- Time-based random access (#80)
- Throw original exceptions on crashes (#87)
- Check for corrupted data (#88)
- FIX: edge cases in processing (#94)
- FIX: prevent saving during time range or fuzzy selection (#89)
- FIX: Workaround for memory leak in single-core mode (#91)
- XENON: Example cuts (#84)
- XENON: proper S1-S2 pairing (#82)
- XENON: Fix pax conversion (#95)
- DOCS: Datastructure docs (#85)

0.4.0 / 2018-08-27
------------------
- S3-protocol I/O (#68, #71, #74)
- Fuzzy matching, context options (#66)
- Fix bug with PyPI lacking MANIFEST (e9771db79bd0c6a148afe1fa8c2ed3d13495da88)
- Zenodo badge (#58)

0.3.0 / 2018-08-13
------------------
- Storage frontend/backend split, several fixes (#46)
- XENON: pax conversion fix (#47)
- Globally configurable mailbox settings (#55, #57)

0.2.0 / 2018-07-03
------------------
- Start documentation
- `ParallelSourcePlugin` to better distribute low-level processing over multiple cores
- `OverlapWindowPlugin` to simplify algorithms that look back and ahead in the data
- Run-dependent config defaults
- XENON: Position reconstruction (tensorflow NN) and corrections

0.1.2 / 2018-05-09
------------------
- Failed to make last patch release.

0.1.1 / 2018-05-09
------------------
- `#19`: list subpackages in setup.py, so numba can find cached code
- Autodeploy from Travis to PyPI
- README badges

0.1.0 / 2018-05-06
------------------
- Initial release