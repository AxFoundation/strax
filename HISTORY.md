1.1.7 / 2022-02-16
---------------------
- Fix savewhen issues (#648)
- Fix testing from straxen (#650)
- Small fix superruns define run (#651)


1.1.6 / 2022-02-03
---------------------
- Extend search field to also print occurrences (#638)
- Extend save when (#631)
- test straxen for coverage and backward compatibility (#635)
- Failing example for test_conditional_entropy (#544)
- python 3.10 (#630)
- deprecate py3.6 py3.7 (#636)
- remove deprecated function (#632)
- Numba 0.55 (#634)
  

1.1.5 / 2022-01-10
---------------------
- StorageFrontend remoteness attribute and test (#617)
- fix line endings (#618)
- Bump numpy (#627)
- Don't cache `hitlet_properties` (#616)


1.1.4 / 2021-12-16
---------------------
- Make truly HDR (#613)
- Remove tight coincidence channel from data_type (#614) 


1.1.3 / 2021-12-13
---------------------
-  Add mode and tags to superrun. (#593) 
-  cache deps (#595) 
-  Fix online monitor bug for only md stored (#596) 
-  speedup get_source with lookupdict (#599)
-  remove config warning and infer_dtype=False (#600)
-  Require pymongo 3.* (#611) 


1.1.2 / 2021-11-19
---------------------
- Descriptor configs (#550)
- Add funcs for getting stored source (#590)
- use logged warnings (#582)
- Fail for - run_ids (#567)
- Infer type from default value in Option (#569, #586, #587)
- Fix buffer issue in highest density regions, adds tests (#591)
- Fix memory usage multi runs (#581)
- Update CONTRIBUTING.md (#585)

Notes:
 - PRs #569, #586, #587 may cause a lot of warnings for options
 

1.1.1 / 2021-10-27
---------------------
- Fix memory leak (#561)
- Fix superrun creation (#562)
- Add deregister plugins (#560)
- Script for testing python setup.py install (#557)


1.1.0 / 2021-10-15
---------------------
major / minor:

- Fix hitlet splitting (#549)
- Add tight channel (#551) 

patch:

- Add read by index plus some extra checks (#529)
- Add drop column option (#530)
- Remove context.apply_selection (#531)
- Add option to support superruns for storage frontends. Adds test (#532)
- Fix issue #536 (#537) 
- Two pbar patches (#538)
- Add get_zarr method to context (#540) 
- Broken metadata error propagation (#541)
- few tests for MongoStorage frontend (#542)
- Fix caching (#545) 
- dds information about failing chunk (#548)
- remove rucio (#552) 
- Allow loading SaveWhen.EXPLICIT time range selection (#553) 
- Changes to savewhen behavior (#554)


1.0.0 / 2021-09-1
---------------------
major / minor:

- Fixing peaklet baseline bias (#486)
- Fix double dependency (#512)

patch:

- Parallel plugin timeout (#489)
- Added pytest.ini (#492)
- Fix nveto processing (#491)
- disable testing py3.6 (#505)
- Fix peaks merging (#506)
- Added export (#508)
- Simplify get_components (#510)
- Allow creation and storing of superruns if SaveWhen > 0 (#509)
- re-use instance of plugin for multi output (#516)
- Add raise if instance are not equal (#517)


0.16.1 / 2021-07-16
---------------------
- Cached lineage if per run default is not allowed (#483, #485)
- Fix define runs and allow storing of superruns (#472, #488)
- Change default pbar behavior (for multiple runs) (#480)
- Reduce numpy warnings (#481, #484)
- Reduce codefactor (#487)


0.16.0 / 2021-06-23
---------------------
- Add select index to compute width (#465)
- int blaseline (#464)
- Fix #452 assert there is a mailbox for the final generator (#463)
- Document fuzzy-for and documentation itself (#471)
- Re ordered time field in cut plugins (#473)
- Github actions for CI (#454, #460)
- Remove travis for testing (#474)
- Remove outdated files/configs (#462)
- Remove overwrite from options (#467)

 
0.15.3 / 2021-06-03
---------------------
- Match cached buffer chunk start times OverlapWindowPlugin (#450)
- Prevent late creation of unattended mailboxes (#452)
- Temporary patch div/zero in hitlets (#447)
- Relax zstd requirements again (#443)
- Don't ignore if pattern also startswith underscore (#445)
- kB/s pbar (#449)


0.15.2 / 2021-05-20
---------------------
- Speed up run selection by ~100x for fixed defaults (#440)
- Use zstd for from base-env for testing (#441)
- Add MB/s pbar (#442)


0.15.1 / 2021-05-04
---------------------
- Refactor hitlets (#430, #436)
- Update classifiers for pipy #437
- Allow Py39 in travis tests (#427) 

0.15.0 / 2021-04-16
---------------------
- Use int32 for peak dt, fix #397 (#403, #426)
- max peak duration (#420)
- Loopplugin touching windows + plugin documentation (#424)
- Move apply selection from context to utils (#425)
- Context testing functions + copy_to_frontend documented (#423)
- Apply function to data & test (#422)

0.14.0 / 2021-04-09
---------------------
- Check data availability for single run (#416) 

0.13.11 / 2021-04-02
---------------------
- Allow re-compression at copy to frontend (#407)
- Bug fix, in processing hitlets (#411)
- Cleanup requirements for boto3 (#414)

0.13.10 / 2021-03-24
---------------------
- Allow multiple targets to be computed simultaneously (#408, #409)
- Numbafy split by containment (#402)
- Infer start/stop from any dtype (#405)
- Add property provided_dtypes to Context (#404)
- Updated OverlapWindowPlugin docs (#401)

0.13.9 / 2021-02-22
---------------------
- Clip progress progressbar (#399)

0.13.8 / 2021-02-09
---------------------
- Specify saver timeout at context level (#394)
- Allow dynamic function creation for dtype copy (#395)
- Close inlined savers on exceptions in multiprocessing (#390)
- Allow option to be overwritten to allow subclassing (#392)
- Fix availability checks (#389)
- Don't print for temp plugins (#391)

0.13.7 / 2021-01-29
---------------------
- Warn for non saved plugins in run selection (#387)
- Cleanup progressbar (#386)

0.13.4 / 2021-01-22
---------------------
- Nveto changes + highest density regions (#384) 
- Parse requirements for testing (#383)
- Added keep_columns into docstring (#382)
- remove slow operators from mongo storage (#382)

0.13.3 / 2021-01-15
---------------------
- Better online monitor queries (#375)
- Multiprocess fix (#376)
- Bugfix (#377)

0.13.2 / 2021-01-04
---------------------
- Speed up st.select_runs by ~100x (#371)
- Finding latest OnlineMonitor data (#374)

0.13.1 / 2020-12-21
---------------------
- Fix bug in baselining (#367)

0.12.7 / 2020-12-21
---------------------
- Fix for select_runs with Nones(#370)
- Numpy requirement fix (#369)
- Documentation maintenance (cad6bce8, 9c023b0d)

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
