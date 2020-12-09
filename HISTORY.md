0.12.6 / 2020-12-09
---------------------
- Muveto + hitlet fixes (#355)
- Add more tests to strax (#359)
- More clear print statement (#362)
- Fix reproducibility of peak split test (#363)
- Fix numpy deps (#364)

0.12.5 / 2020-12-06
---------------------
- Finally fix time selection bug (#345)
- Allow multioutput for loop plugin (#357)
- Allow copy from frontend to frontend (#351)
- Add more tests to strax (#359)
- Small bugfix progressbar (#353)
- Smooth database initiation CMT (#356)
- Remove s3 storage (#358)

0.12.4 / 2020-11-13
---------------------
- Limit mongo backend memory usage (#349)
- Small CMT simplification (#347)

0.12.3 / 2020-11-05
---------------------
- Updates to mongo.py (#335, #336 )
- Several bug-fixes (#340, #343, #344, #338)
- Contributions to documentation (#342, #344)
- Accommodate  scada (#318)

0.12.2 / 2020-10-15
---------------------
- OnlineMonitor in mongo.py (#315, #323)
- Several bugfixes (#320, #321, #327, #329, #334)
- Option to give range to sum_waveform (#322)


0.12.1 / 2020-09-10
---------------------
- Added the "Corrections Management Tool" (#303)
- Check of loop dependency for multioutput plugins (#312)
- Fix memory leak peaksplitting (#309)


0.12.0 / 2020-08-17
---------------------
- Add backend for rucio (#300)
- Hitlets data types and utilities (#275)
- Apply function to data prior to delivering (#304)
- Child options for inherited plugins (#297)
- Introducing a template for pull-requests (#302)
- Make fuzzy_for option more intuitive (#306)


0.11.3 / 2020-07-29
---------------------
- bugfix in new progressbar feature (#296)


0.11.2 / 2020-07-21
---------------------
- new plugin type: CutPlugin (#274)
- add progressbar (#276)
- allow for plugin-specific chunk-sizes (#277)
- broaden scope of endtime check in chunk.py (#281)
- change dtype of saturated channels (#286)
- several (bug-)fixes (#289, #288, #284, #278, #273)


0.11.1 / 2020-04-28
---------------------
- Per-datatype rechunking control (#272)
- Check defaults are consistent across plugins (#271)
- Documentation / comment updates (#269, #268)
- Peak splitter test (#267)
- Fix bug in pulse_processing when not flipping waveforms (#266)
- Fix lineage hash caching (#263)
- Flexible run start time estimation (905335)


0.11.0 / 2020-04-28
---------------------
- `accumulate` method for basic out-of-core processing (#253)
- Lone hit integration routines (#261)
- Record amplitude bit shift, fix saturation counting (#260)
- Make index_of_fraction more reusable (#257)
- DataDirectory does not `deep_scan` or `provide_run_metadata` by default
- Numba 0.49 compatibility


0.10.3 / 2020-04-13
-------------------
- Disable timeout / make it 24 hours (#255)
- Minor fixes for warning messages


0.10.2 / 2020-04-06
--------------------
- Fix loading of incomplete data (#251)
- Fx exception handling (#252)
- Fix hitfinder buffer overrun if too few thresholds specified (bc2c2b)
- keep_columns selection option (4e2550)
- Assume all run metadata is in UTC (4e223e)
- Can now specify `*` in forbid_creation_of (86552f)
- Simplify length computations (#250)


0.10.1 / 2020-03-18
--------------------
- Even lazier processing (#248)
- Fix multiprocessing bug for multi-output plugins (0f1b1d, 1e388a)
- Make baselining work in non-jitted mode (8f1f23)
- Better exception handling in estimate_run_start (9e2f88, #249)
- Disable run defaults by default (c1f094)


0.10.0 / 2020-03-15
------------------
- Lazy mailbox for processing (#241)
- Baselining checks for nonzero first fragment (#243)
- Add `size_mb` context method
- Force time range to be integer
- Update messages and exceptions (#244, #245)

0.9.1 / 2020-03-08
------------------
- Fix bug in input synchronization

0.9.0 / 2020-03-05
------------------
- Use chunks with defined start/end rather than plain arrays (#235)
- Mandate time fields in all datatypes (#235)
- Remove unnecessary fields from raw-records (#235, #237)
- Allow compute to take start and end fields (#239)
- Channel-dependent hitfinder threshold (#234)
- Wait on Executors during shutdown (#236)
- Protect hitfinder against buffer overruns

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
