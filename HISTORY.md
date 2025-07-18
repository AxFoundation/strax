2.2.1 / 2025-07-18
---------------------
* Check the `data_type` of returned chunk of plugin by @dachengx in https://github.com/AxFoundation/strax/pull/995
* Topological level and shortest dependency path of dependency tree (directed acyclic graph) by @dachengx in https://github.com/AxFoundation/strax/pull/996
* Do not assign `_multi_output_topics` when the topic is from a loader by @dachengx in https://github.com/AxFoundation/strax/pull/997

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.2.0...v2.2.1



2.2.0 / 2025-05-16
---------------------
* Fix a corner case where the chunk size is zero by @dachengx in https://github.com/AxFoundation/strax/pull/988
* Do not initialize any saver when dropping columns by @dachengx in https://github.com/AxFoundation/strax/pull/989
* Add more comments about `merge_arrs` by @dachengx in https://github.com/AxFoundation/strax/pull/991
* Use numbered version of `docformatter` by @dachengx in https://github.com/AxFoundation/strax/pull/992
* Constrain numcodecs to be less than 0.16.0 by @dachengx in https://github.com/AxFoundation/strax/pull/993

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.1.4...v2.2.0


2.1.4 / 2025-03-09
---------------------
* Allow changing chunk size and compressor in `merge_per_chunk_storage` by @dachengx in https://github.com/AxFoundation/strax/pull/986

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.1.3...v2.1.4


2.1.3 / 2025-03-02
---------------------
* Prevent infinite loop in `get_splits` by @dachengx in https://github.com/AxFoundation/strax/pull/983
* All copy data from multiple per-chunk storage by @dachengx in https://github.com/AxFoundation/strax/pull/984

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.1.2...v2.1.3


2.1.2 / 2025-02-24
---------------------
* Allow `depends_on` rechunking by @dachengx in https://github.com/AxFoundation/strax/pull/975
* Save memory when decompressing data by @dachengx in https://github.com/AxFoundation/strax/pull/976
* Use `zstandard` to decompress chunks to save memory by @dachengx in https://github.com/AxFoundation/strax/pull/977
* Prevent unnecessary reference to chunks by @dachengx in https://github.com/AxFoundation/strax/pull/979
* Add `DECOMPRESS_BUFFER_SIZE` by @dachengx in https://github.com/AxFoundation/strax/pull/980

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.1.1...v2.1.2


2.1.1 / 2025-02-13
---------------------
* Clip `center_time` to be within the `time` and `endtime` by @dachengx in https://github.com/AxFoundation/strax/pull/973

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.1.0...v2.1.1


2.1.0 / 2025-02-03
---------------------
* Add a flag indicating whether check subruns' configs by @dachengx in https://github.com/AxFoundation/strax/pull/964
* Revert "Add a flag indicating whether check subruns' configs" by @dachengx in https://github.com/AxFoundation/strax/pull/965
* Skip continuity check when using `ExhaustPlugin` by @dachengx in https://github.com/AxFoundation/strax/pull/966
* Allow two-sided overlapping window by @dachengx in https://github.com/AxFoundation/strax/pull/970
* Push the latest change by @dachengx in https://github.com/AxFoundation/strax/pull/971

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.5...v2.1.0


2.0.5 / 2025-01-24
---------------------
* When loader is available, do not save anything by @dachengx in https://github.com/AxFoundation/strax/pull/953
* Do not cache `find_split_points` by @dachengx in https://github.com/AxFoundation/strax/pull/955
* Set `chunk_number` as attribute of `Plugin` to pass `chunk_i` by @dachengx in https://github.com/AxFoundation/strax/pull/956
* First and last channel inside peak(let)s by @dachengx in https://github.com/AxFoundation/strax/pull/954
* Temporary plugin should keep order of targets by @dachengx in https://github.com/AxFoundation/strax/pull/958
* Some times the sum of data is zero due to numerical inaccuracy by @dachengx in https://github.com/AxFoundation/strax/pull/959
* Use `base` of dtype in `set_nan_defaults` by @dachengx in https://github.com/AxFoundation/strax/pull/960
* Remove `CorrectionsInterface` by @dachengx in https://github.com/AxFoundation/strax/pull/961
* Drop python 3.9 and loosen requirement of `numpy` by @dachengx in https://github.com/AxFoundation/strax/pull/962

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.4...v2.0.5


2.0.4 / 2025-01-13
---------------------
* Numbafy `merge_peaks` by @dachengx in https://github.com/AxFoundation/strax/pull/946
* Propagate min/max diff in merged S2s by @WenzDaniel in https://github.com/AxFoundation/strax/pull/937
* Fix bug of numbafied `merge_peaks` by @dachengx in https://github.com/AxFoundation/strax/pull/947
* Fix `merge_peaks` by removing `endtime` assignment by @dachengx in https://github.com/AxFoundation/strax/pull/948
* Do not concatenate empty dataframes by @dachengx in https://github.com/AxFoundation/strax/pull/949
* Collect `endtime` from `_merge_peaks` function by @dachengx in https://github.com/AxFoundation/strax/pull/950
* `OverlapWindowPlugin` support multiple outputs by @dachengx in https://github.com/AxFoundation/strax/pull/951

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.3...v2.0.4


2.0.3 / 2024-12-27
---------------------
* Move `set_nan_defaults` from straxen to strax by @dachengx in https://github.com/AxFoundation/strax/pull/936
* Move `compute_center_times` from straxen to strax by @dachengx in https://github.com/AxFoundation/strax/pull/938
* First convert to int then add float to keep precision by @dachengx in https://github.com/AxFoundation/strax/pull/939
* Use `bool` instead of `np.bool_` by @dachengx in https://github.com/AxFoundation/strax/pull/940
* Calculate `area_fraction_top`, `center_time` and `median_time` at peaklet level by @dachengx in https://github.com/AxFoundation/strax/pull/941
* Delete input chunks after compute method to save memory by @dachengx in https://github.com/AxFoundation/strax/pull/942
* Reduce RAM usage of `find_hit_integration_bounds` by @dachengx in https://github.com/AxFoundation/strax/pull/943

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.2...v2.0.3


2.0.2 / 2024-11-19
---------------------
* Raise error when peaks overlapping in `merge_peaks` by @dachengx in https://github.com/AxFoundation/strax/pull/927
* Print progress bar in `dry_load_files` by @dachengx in https://github.com/AxFoundation/strax/pull/928
* Fix a small bug in `merge_arrs` by @dachengx in https://github.com/AxFoundation/strax/pull/930

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.1...v2.0.2


2.0.1 / 2024-11-14
---------------------
* Allow `chunk_number` to be list or tuple in `dry_load_files` function by @dachengx in https://github.com/AxFoundation/strax/pull/921
* Fixing store_data_start in recursive peak splitter calls and peaklet["length'] fix in store_downsampled_waveform by @HenningSE in https://github.com/AxFoundation/strax/pull/920
* Run garbage collection after yield result in `Plugin.iter` by @dachengx in https://github.com/AxFoundation/strax/pull/922
* Add a function to get size of a single item of data_type in bytes by @dachengx in https://github.com/AxFoundation/strax/pull/923
* Speed up `get_dependencies` by @dachengx in https://github.com/AxFoundation/strax/pull/924
* Add more kwargs to `dry_load_files` by @dachengx in https://github.com/AxFoundation/strax/pull/925
* Add enforcement for `np.sort` and `np.argsort` by @yuema137 in https://github.com/AxFoundation/strax/pull/918

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v2.0.0...v2.0.1


2.0.0 / 2024-10-18
---------------------
* Allow `_chunk_number` to be list or tuple by @dachengx in https://github.com/AxFoundation/strax/pull/856
* Filter out duplicated targets in function `get_iter` by @dachengx in https://github.com/AxFoundation/strax/pull/860
* Check duplicate `depends_on` of `Plugin` by @yuema137 in https://github.com/AxFoundation/strax/pull/859
* Set `lineage` and `lineage_hash` simultaneously by @dachengx in https://github.com/AxFoundation/strax/pull/861
* Save run metadata in better format by @dachengx in https://github.com/AxFoundation/strax/pull/868
* Include `chunk_number` in lineage: Per chunk storage by @dachengx in https://github.com/AxFoundation/strax/pull/863
* Allow splitting in rechunking by @dachengx in https://github.com/AxFoundation/strax/pull/865
* Add subruns information in `DataKey` of superruns to track metadata by @dachengx in https://github.com/AxFoundation/strax/pull/866
* Use `pyproject.toml` to install strax by @dachengx in https://github.com/AxFoundation/strax/pull/870
* Save subruns information of hyperrun by @dachengx in https://github.com/AxFoundation/strax/pull/869
* Unify functionality of super and hyperrun by @dachengx in https://github.com/AxFoundation/strax/pull/871
* Add more tests about `PostOffice` and `get_components` by @dachengx in https://github.com/AxFoundation/strax/pull/872
* Prohibit `chunk_number` for `LoopPlugin` and `OverlapWindowPlugin` by @dachengx in https://github.com/AxFoundation/strax/pull/877
* Only save combined `data_type` in only-combining mode by @dachengx in https://github.com/AxFoundation/strax/pull/878
* Remove `get_meta` function from Context by @dachengx in https://github.com/AxFoundation/strax/pull/879
* Remove redundant spaces by @dachengx in https://github.com/AxFoundation/strax/pull/881
* No `run_id` dependent plugin version by @dachengx in https://github.com/AxFoundation/strax/pull/880
* Show warning when the `chunk_number` is not needed by @dachengx in https://github.com/AxFoundation/strax/pull/883
* Refactor nv plugins by @WenzDaniel in https://github.com/AxFoundation/strax/pull/744
* Use `run_id_output` sorting `final_result` in `multi_run` by @dachengx in https://github.com/AxFoundation/strax/pull/885
* `_base_hash_on_config` should not be an attribute by @dachengx in https://github.com/AxFoundation/strax/pull/882
* Add `combining` into the `DataKey` by @dachengx in https://github.com/AxFoundation/strax/pull/886
* Minor debug for the `pyproject.toml` by @dachengx in https://github.com/AxFoundation/strax/pull/888
* Fix the usage of scripts by @dachengx in https://github.com/AxFoundation/strax/pull/890
* Deprecate `selection_str` by @dachengx in https://github.com/AxFoundation/strax/pull/891
* Add `run_id`  independent function to get the dependencies `datat_type` by @dachengx in https://github.com/AxFoundation/strax/pull/892
* Allow get_df on all data_types by @lorenzomag in https://github.com/AxFoundation/strax/pull/887
* Select targeted software frontend in a clever way by @dachengx in https://github.com/AxFoundation/strax/pull/893
* Cancel usage of `chunk_number` if loading the whole dependency by @dachengx in https://github.com/AxFoundation/strax/pull/894
* Add function of dependency level of `data_types` by @dachengx in https://github.com/AxFoundation/strax/pull/896
* Add option to save first samples of peak(lets) waveform by @HenningSE in https://github.com/AxFoundation/strax/pull/867
* Set single thread of zstd and blosc by @dachengx in https://github.com/AxFoundation/strax/pull/899
* Set default `max_downsample_factor_waveform_start` as not `None` by @dachengx in https://github.com/AxFoundation/strax/pull/900
* Turn back to `zstd` because `zstandard` raise errors by @dachengx in https://github.com/AxFoundation/strax/pull/902
* Do not add producer that has been added by saver by @dachengx in https://github.com/AxFoundation/strax/pull/901
* Propagate n_top_channels to _add_lone_hits by @HenningSE in https://github.com/AxFoundation/strax/pull/907
* Add option to merge `lone_hits` into `data_start` by @HenningSE in https://github.com/AxFoundation/strax/pull/908
* Save `data_start` even there is no downsampling by @dachengx in https://github.com/AxFoundation/strax/pull/909
* Remove `max_downsample_factor_waveform_start`, simplify the saving of `data_start` by @dachengx in https://github.com/AxFoundation/strax/pull/910
* Make variables names more robust in `peak_dtype` by @dachengx in https://github.com/AxFoundation/strax/pull/911
* Set `SingleThreadProcessor` as the default processor by @dachengx in https://github.com/AxFoundation/strax/pull/904
* Switch to master for docformatter by @dachengx in https://github.com/AxFoundation/strax/pull/912
* Add `max_time` which is the time when hit reaches its maximum by @dachengx in https://github.com/AxFoundation/strax/pull/913
* Small fix of dtype description by @dachengx in https://github.com/AxFoundation/strax/pull/914
* Exclude git repo from the package metadata for PyPI by @dachengx in https://github.com/AxFoundation/strax/pull/915

New Contributors
* @yuema137 made their first contribution in https://github.com/AxFoundation/strax/pull/859
* @lorenzomag made their first contribution in https://github.com/AxFoundation/strax/pull/887
* @HenningSE made their first contribution in https://github.com/AxFoundation/strax/pull/867

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.5...v2.0.0


1.6.5 / 2024-07-10
---------------------
* Single threaded alternative processor by @JelleAalbers in https://github.com/AxFoundation/strax/pull/773
* Patch sharedarray version due to numpy updating by @dachengx in https://github.com/AxFoundation/strax/pull/850
* Fix cache of Rechunker, zero chunk is not None by @dachengx in https://github.com/AxFoundation/strax/pull/849
* Print writable storage by @dachengx in https://github.com/AxFoundation/strax/pull/851
* Save multiple output when iteration stops in `PostOffice` by @dachengx in https://github.com/AxFoundation/strax/pull/852
* Do not cache numba decorated `split_array` by @dachengx in https://github.com/AxFoundation/strax/pull/854

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.4...v1.6.5


1.6.4 / 2024-05-27
---------------------
* Minor fix on `_is_superrun` variable  names by @dachengx in https://github.com/AxFoundation/strax/pull/837
* Implement Hyperrun by @dachengx in https://github.com/AxFoundation/strax/pull/838
* Add function to collect `data_type` and `data_kind` by @dachengx in https://github.com/AxFoundation/strax/pull/839
* Check `include_tags` and `exclude_tags` by @dachengx in https://github.com/AxFoundation/strax/pull/841

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.3...v1.6.4


1.6.3 / 2024-04-30
---------------------
* Install `graphviz` for the pytests by @dachengx in https://github.com/AxFoundation/strax/pull/817
* Increase the timing precision of progress bar by @dachengx in https://github.com/AxFoundation/strax/pull/819
* Initialize plugin because `depends_on` can be property by @dachengx in https://github.com/AxFoundation/strax/pull/820
* Update context.py by @WenzDaniel in https://github.com/AxFoundation/strax/pull/821
* Disable tqdm progress bar when `check_available` is empty by @dachengx in https://github.com/AxFoundation/strax/pull/822
* Check the consistency of number of items in metadata and data in `dry_load_files` function by @dachengx in https://github.com/AxFoundation/strax/pull/824
* Remove `strax.plugin` by @dachengx in https://github.com/AxFoundation/strax/pull/825
* Pick out selection applying function by @dachengx in https://github.com/AxFoundation/strax/pull/826
* Add `CutList` by @dachengx in https://github.com/AxFoundation/strax/pull/827
* Update tags handling, added comment field. Allows to define superuns … by @WenzDaniel in https://github.com/AxFoundation/strax/pull/798
* Prevent start being negative by @dachengx in https://github.com/AxFoundation/strax/pull/828
* Tiny change on the trailing space by @dachengx in https://github.com/AxFoundation/strax/pull/830
* Add `register_cut_list` by @dachengx in https://github.com/AxFoundation/strax/pull/831
* Record all base classes when multiple inheritance by @dachengx in https://github.com/AxFoundation/strax/pull/832
* Multiple output `DownChunkingPlugin` by @dachengx in https://github.com/AxFoundation/strax/pull/833
* Add `ExhaustPlugin` that exhausts all chunks when fetching data by @dachengx in https://github.com/AxFoundation/strax/pull/835

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.2...v1.6.3


1.6.2 / 2024-04-04
---------------------
* Use parentheses to separate the class name and attributes in the representation of StorageFrontend by @dachengx in https://github.com/AxFoundation/strax/pull/809
* Specifically install `lxml_html_clean` by @dachengx in https://github.com/AxFoundation/strax/pull/812
* Add a function to purge unused configs by @dachengx in https://github.com/AxFoundation/strax/pull/800
* Warn if user checks is_stored for plugin not always saved by @cfuselli in https://github.com/AxFoundation/strax/pull/796
* Bump urllib3 from 2.2.0 to 2.2.1 in /extra_requirements by @dependabot in https://github.com/AxFoundation/strax/pull/808
* Do not call `get_components` in `is_stored` by @dachengx in https://github.com/AxFoundation/strax/pull/813

New Contributors
* @cfuselli made their first contribution in https://github.com/AxFoundation/strax/pull/796

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.1...v1.6.2


1.6.1 / 2024-02-17
---------------------
* Remove a redundant function and fix some style by @dachengx in https://github.com/AxFoundation/strax/pull/795
* Find the frontends which stored the targets by @dachengx in https://github.com/AxFoundation/strax/pull/802
* Simpler chunk length check, avoid recursion limit crash by @JelleAalbers in https://github.com/AxFoundation/strax/pull/803
* Deprecate the usage of `XENONnT/ax_env` by @dachengx in https://github.com/AxFoundation/strax/pull/804
* Add a function to directly load file from strax folder by @dachengx in https://github.com/AxFoundation/strax/pull/801


**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.6.0...v1.6.1


1.6.0 / 2024-01-15
---------------------
* `np.float` is deprecated by @dachengx in https://github.com/AxFoundation/strax/pull/789
* Update pymongo and solve the error of pytest by @dachengx in https://github.com/AxFoundation/strax/pull/791


**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.5...v1.6.0


1.5.5 / 2023-12-21
---------------------
* Update readthedocs configuration by @dachengx in https://github.com/AxFoundation/strax/pull/763
* Proposal to use pre-commit for continuous integration by @dachengx in https://github.com/AxFoundation/strax/pull/762
* Update authorship and copyright info by @JelleAalbers in https://github.com/AxFoundation/strax/pull/771
* Guard hitlet entropy test from numerical errors by @JelleAalbers in https://github.com/AxFoundation/strax/pull/772
* Deregister partially replaced multi-output plugins by @JelleAalbers in https://github.com/AxFoundation/strax/pull/775
* Fix caching issue by @WenzDaniel in https://github.com/AxFoundation/strax/pull/768
* Add chunk yielding plugin and tests by @WenzDaniel in https://github.com/AxFoundation/strax/pull/769
* Avoid deprecated generated_jit by @JelleAalbers in https://github.com/AxFoundation/strax/pull/784
* Also copy dps and remove redundant checks. by @WenzDaniel in https://github.com/AxFoundation/strax/pull/777
* Add hot fix for copy_to_buffer by @WenzDaniel in https://github.com/AxFoundation/strax/pull/785
* Upgrade compare-metadata function by @KaraMelih in https://github.com/AxFoundation/strax/pull/778
* Add warning by @WenzDaniel in https://github.com/AxFoundation/strax/pull/776

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.4...v1.5.5


1.5.4 / 2023-09-19
---------------------
* Split compare_metadata into utils.compare_meta by @dachengx in https://github.com/AxFoundation/strax/pull/754
* Change endtime - time >= 0 to endtime >= time by @JYangQi00 in https://github.com/AxFoundation/strax/pull/756
* Mandatorily wrap `_read_chunk` in a `check_chunk_n` decorator by @dachengx in https://github.com/AxFoundation/strax/pull/758

New Contributors
* @JYangQi00 made their first contribution in https://github.com/AxFoundation/strax/pull/756

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.3...v1.5.4


1.5.3 / 2023-08-29
---------------------
* Add small selection functions by @WenzDaniel in https://github.com/AxFoundation/strax/pull/746
* Patch plugin cache by @WenzDaniel in https://github.com/AxFoundation/strax/pull/748
* Update version of urllib3, remove version control of deepdiff by @dachengx in https://github.com/AxFoundation/strax/pull/749
* Check chunk size right after loading chunk by @dachengx in https://github.com/AxFoundation/strax/pull/752


**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.2...v1.5.3


1.5.2 / 2023-07-06
---------------------
* Use warning also in `abs_time_to_prev_next_interval` by @dachengx in https://github.com/AxFoundation/strax/pull/738


**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.1...v1.5.2


1.5.1 / 2023-06-22
---------------------
* Fix argsort inside numba.jit using kind='mergesort' by @dachengx in https://github.com/AxFoundation/strax/pull/721
* Fix urllib3 version to 1.26.15 by @dachengx in https://github.com/AxFoundation/strax/pull/723
* Save other fields in the merged peaks to their default value by @dachengx in https://github.com/AxFoundation/strax/pull/722
* add a metadata comparison method by @KaraMelih in https://github.com/AxFoundation/strax/pull/706
* Accelerate select_runs by @shenyangshi in https://github.com/AxFoundation/strax/pull/727
* Stop assigning dependabot to Joran by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/732
* Bump urllib3 from 1.26.15 to 2.0.2 in /extra_requirements by @dependabot in https://github.com/AxFoundation/strax/pull/729
* Add new general fucntion which computes dt to some interval by @WenzDaniel in https://github.com/AxFoundation/strax/pull/726
* Check whether `things` and `containers` are sorted by @dachengx in https://github.com/AxFoundation/strax/pull/725
* Set start of further chunk to be the smallest start of dependencies by @dachengx in https://github.com/AxFoundation/strax/pull/715
* Fix touching window by @dachengx in https://github.com/AxFoundation/strax/pull/736

New Contributors
* @KaraMelih made their first contribution in https://github.com/AxFoundation/strax/pull/706
* @shenyangshi made their first contribution in https://github.com/AxFoundation/strax/pull/727

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.5.0...v1.5.1


1.5.0 / 2023-05-02
---------------------
* Fix ipython version by @dachengx in https://github.com/AxFoundation/strax/pull/719
* Do not change channel when sort_by_time by @dachengx in https://github.com/AxFoundation/strax/pull/718
* Save hits level information(hits time difference) in peaks by @dachengx in https://github.com/AxFoundation/strax/pull/716

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.4.3...v1.5.0


1.4.3 / 2023-04-22
---------------------
* Select max gaps from positive gaps by @dachengx in https://github.com/AxFoundation/strax/pull/708

New Contributors
* @dachengx made their first contribution in https://github.com/AxFoundation/strax/pull/708

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.4.2...v1.4.3


1.4.2 / 2023-03-08
---------------------
* Patch md access in the rechunker by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/711
* Raise compression errors if unable by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/714

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.4.1...v1.4.2


1.4.1 / 2023-02-13
---------------------
* Rechunker using Mailbox by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/710

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.4.0...v1.4.1


1.4.0 / 2022-10-17
---------------------
* Add capability for building summed waveform over channel subset by @DCichon in https://github.com/AxFoundation/strax/pull/565
* Fixed delta peak timestamp problem by @FaroutYLq in https://github.com/AxFoundation/strax/pull/702

Notes
 -  Breaking changes in the peak-building chain due to #565
New Contributors
 - @DCichon made their first contribution in https://github.com/AxFoundation/strax/pull/565

**Full Changelog**: https://github.com/AxFoundation/strax/compare/v1.3.0...v1.4.0


1.3.0 / 2022-09-09
---------------------
* Restructure plugins by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/700
* Numpy caching of data in online monitor storage by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/698
* Fix overflow bug in sort_by_time add little test by @WenzDaniel in https://github.com/AxFoundation/strax/pull/695
* Refactor in preparation for PyMongo 4.0 by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/656
* Plugin log property by @jmosbacher in https://github.com/AxFoundation/strax/pull/588


1.2.3 / 2022-06-07
---------------------
* Prevent float/int funkyness in hitlet processing by @JoranAngevaare in https://github.com/AxFoundation/strax/pull/694


1.2.2 / 2022-05-11
---------------------
- Add option to ignore errors in multirun loading (#653)
- Auto version, fix #217 (#689)
- Add basics documentation - split Config and Plugin docs (#691)
- Add n_hits comment in code (#692)
- Rechunker script (#686)


1.2.1 / 2022-04-12
---------------------
- run dependabot remotely (#683)
- Docs fixing (#684)
- Allow different chunk size (#687)


1.2.0 / 2022-03-09
---------------------
- Added lone hit area to area per channel (#649)

1.1.8 / 2022-03-08
---------------------
- Fix saving behavior of multioutput plugins with different SaveWhens (#674)
- Change tempdirs in test (#657)
- Define extra kwargs based on cut_by (db14f809414fe91c4e16d04bd7f166970891e591)
- Update run_selection.py (#658)
- Ignore raises on testing (#669)
- Documentation tweaks (#670)
- Test for inline plugin (#673)


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
