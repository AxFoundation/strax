import glob
import shutil
import tempfile
import os
import os.path as osp
import pytest

from strax import RUN_METADATA_PATTERN
from strax.testutils import *

processing_conditions = pytest.mark.parametrize(
    "allow_multiprocess,max_workers,processor",
    [
        (False, 1, "threaded_mailbox"),
        (True, 2, "threaded_mailbox"),
        (False, 1, "single_thread"),
    ],
)


@processing_conditions
def test_core(allow_multiprocess, max_workers, processor):
    mystrax = strax.Context(
        storage=[],
        register=[Records, Peaks],
        processors=[processor],
        allow_multiprocess=allow_multiprocess,
        use_per_run_defaults=True,
    )
    bla = mystrax.get_array(run_id=run_id, targets="peaks", max_workers=max_workers)
    p = mystrax.get_single_plugin(run_id, "records")
    assert len(bla) == p.config["recs_per_chunk"] * p.config["n_chunks"]
    assert bla.dtype == strax.peak_dtype()


@processing_conditions
def test_core_df(allow_multiprocess, max_workers, processor, caplog):
    """Test that get_df works with N-dimensional data."""
    """Test that get_df works with N-dimensional data."""
    mystrax = strax.Context(
        storage=[],
        register=[Records, Peaks],
        processors=[processor],
        allow_multiprocess=allow_multiprocess,
        use_per_run_defaults=True,
    )

    df = mystrax.get_df(run_id=run_id, targets="peaks", max_workers=max_workers)
    p = mystrax.get_single_plugin(run_id, "records")
    assert len(df.loc[0, "data"]) == 200
    assert len(df) == p.config["recs_per_chunk"] * p.config["n_chunks"]
    assert (
        "contain non-scalar entries. Some pandas functions (e.g., groupby, apply)"
        " might not perform as expected on these columns." in caplog.text
    )


def test_post_office_state():
    mystrax = strax.Context(
        storage=[],
        register=[Records, Peaks],
        use_per_run_defaults=True,
    )
    components = mystrax.get_components(run_id, "peaks")
    processor = strax.PROCESSORS["single_thread"](components)
    processor.post_office.state()


def test_multirun():
    for max_workers in [1, 2]:
        mystrax = strax.Context(
            storage=[],
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )
        bla = mystrax.get_array(run_id=["0", "1"], targets="peaks", max_workers=max_workers)
        p = mystrax.get_single_plugin(run_id, "records")
        n = p.config["recs_per_chunk"] * p.config["n_chunks"]
        assert len(bla) == n * 2
        np.testing.assert_equal(bla["run_id"], np.array(["0"] * n + ["1"] * n))


@processing_conditions
def test_filestore(allow_multiprocess, max_workers, processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        mystrax = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            processors=[processor],
            allow_multiprocess=allow_multiprocess,
            use_per_run_defaults=True,
        )

        assert not mystrax.is_stored(run_id, "peaks")
        mystrax.scan_runs()
        assert mystrax.list_available("peaks") == []

        # Create it with dropping columns
        mystrax.get_array(run_id=run_id, targets="peaks", keep_columns=["time"])
        assert not mystrax.is_stored(run_id, "peaks")

        # Create it
        peaks_1 = mystrax.get_array(run_id=run_id, targets="peaks")
        p = mystrax.get_single_plugin(run_id, "records")
        assert len(peaks_1) == p.config["recs_per_chunk"] * p.config["n_chunks"]

        assert mystrax.is_stored(run_id, "peaks")
        mystrax.scan_runs()
        assert mystrax.list_available("peaks") == [run_id]
        assert mystrax.scan_runs()["name"].values.tolist() == [run_id]

        # We should have two directories
        data_dirs = sorted(glob.glob(osp.join(temp_dir, "*/")))
        assert len(data_dirs) == 2

        # The first dir contains peaks.
        # It should have one data chunk (rechunk is on) and a metadata file
        prefix = strax.dirname_to_prefix(data_dirs[0])
        assert sorted(os.listdir(data_dirs[0])) == [
            f"{prefix}-000000",
            RUN_METADATA_PATTERN % prefix,
        ]

        # Check metadata got written correctly.
        metadata = mystrax.get_metadata(run_id, "peaks")
        assert len(metadata)
        assert "writing_ended" in metadata
        assert "exception" not in metadata
        assert len(metadata["chunks"]) == 1

        # Check data gets loaded from cache, not rebuilt
        md_filename = osp.join(data_dirs[0], RUN_METADATA_PATTERN % prefix)
        mtime_before = osp.getmtime(md_filename)
        peaks_2 = mystrax.get_array(run_id=run_id, targets="peaks")
        np.testing.assert_array_equal(peaks_1, peaks_2)
        assert mtime_before == osp.getmtime(md_filename)

        # Test the zipfile store. Zipping is still awkward...
        zf = osp.join(temp_dir, f"{run_id}.zip")
        strax.ZipDirectory.zip_dir(temp_dir, zf, delete=True)
        assert osp.exists(zf)

        mystrax = strax.Context(
            storage=strax.ZipDirectory(temp_dir),
            use_per_run_defaults=True,
            register=[Records, Peaks],
        )
        metadata_2 = mystrax.get_metadata(run_id, "peaks")
        assert metadata == metadata_2


def test_datadirectory_deleted():
    """Test deleting the data directory does not cause crashes or silent failures to save (#93)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = osp.join(temp_dir, "bla")
        os.makedirs(data_dir)

        mystrax = strax.Context(
            storage=strax.DataDirectory(data_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )

        # Delete directory AFTER context is created
        shutil.rmtree(data_dir)

        mystrax.scan_runs()
        assert not mystrax.is_stored(run_id, "peaks")
        assert mystrax.list_available("peaks") == []

        mystrax.make(run_id=run_id, targets="peaks")

        mystrax.scan_runs()
        assert mystrax.is_stored(run_id, "peaks")
        assert mystrax.list_available("peaks") == [run_id]


def test_fuzzy_matching():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )

        st.make(run_id=run_id, targets="peaks")

        # Changing option causes data not to match
        st.set_config(dict(base_area=1))
        assert not st.is_stored(run_id, "peaks")
        assert st.list_available("peaks") == []

        # In fuzzy context, data does match
        st2 = st.new_context(fuzzy_for=("peaks",))
        assert st2.is_stored(run_id, "peaks")
        assert st2.list_available("peaks") == [run_id]

        # And we can actually load it
        st2.get_metadata(run_id, "peaks")
        st2.get_array(run_id, "peaks")

        # Fuzzy for options also works
        st3 = st.new_context(fuzzy_for_options=("base_area",))
        assert st3.is_stored(run_id, "peaks")

    # No saving occurs at all while fuzzy matching
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir),
            register=[Records, Peaks],
            use_per_run_defaults=True,
            fuzzy_for=("records",),
        )
        st.make(run_id, "peaks")
        assert not st.is_stored(run_id, "peaks")
        assert not st.is_stored(run_id, "records")


@processing_conditions
def test_exception(allow_multiprocess, max_workers, processor):
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir),
            register=[Records, Peaks],
            processors=[processor],
            allow_multiprocess=allow_multiprocess,
            config=dict(crash=True),
            use_per_run_defaults=True,
        )

        # Check correct exception is thrown
        with pytest.raises(SomeCrash):
            st.make(run_id=run_id, targets="peaks", max_workers=max_workers)

        # Check exception is recorded in metadata
        # in both its original data type and dependents
        for target in ("peaks", "records"):
            assert "SomeCrash" in st.get_metadata(run_id, target)["exception"]

        # Check corrupted data does not load
        st.context_config["forbid_creation_of"] = ("peaks",)
        with pytest.raises(strax.DataNotAvailable):
            st.get_df(run_id=run_id, targets="peaks", max_workers=max_workers)


def test_exception_in_saver(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )

        def kaboom(*args, **kwargs):
            raise SomeCrash

        old_save = strax.save_file
        try:
            strax.save_file = kaboom
            with pytest.raises(SomeCrash):
                st.make(run_id=run_id, targets="records")
        finally:
            strax.save_file = old_save


def test_random_access():
    """Test basic random access
    TODO: test random access when time info is not provided directly
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Hack to enable testing if only required chunks are loaded
        Peaks.rechunk_on_save = False

        st = strax.Context(
            storage=strax.DataDirectory(temp_dir),
            register=[Records, Peaks, PeakClassification],
            use_per_run_defaults=True,
        )

        with pytest.raises(strax.DataNotAvailable):
            # Time range selection requires data already available
            st.get_df(run_id, "peaks", time_range=(3, 5))

        st.make(run_id=run_id, targets=("peaks", "peak_classification"))

        # Second part of hack: corrupt data by removing one chunk
        dirname = str(st.key_for(run_id, "peaks"))
        os.remove(os.path.join(temp_dir, dirname, strax.dirname_to_prefix(dirname) + "-000000"))

        with pytest.raises(FileNotFoundError):
            st.get_array(run_id, "peaks")

        df = st.get_array(run_id, "peaks", time_range=(3, 5))
        # Also test without the progress-bar
        df_pbar = st.get_array(run_id, "peaks", time_range=(3, 5), progress_bar=False)
        p = st.get_single_plugin(run_id, "records")
        assert len(df) == 2 * p.config["recs_per_chunk"]
        assert df["time"].min() == 3
        assert df["time"].max() == 4
        assert np.all(df == df_pbar), "progress-bar changes the result?!?"

        # Try again with unaligned chunks
        df = st.get_array(run_id, ["peaks", "peak_classification"], time_range=(3, 5))
        assert len(df) == 2 * p.config["recs_per_chunk"]
        assert df["time"].min() == 3
        assert df["time"].max() == 4

    # Remove hack
    Peaks.rechunk_on_save = True


def test_rechunk_on_save():
    """Test saving works with and without rechunk on save.

    Doesn't test whether rechunk on save actually rechunks: that's done in test_filestore

    """
    for do_rechunk in (False, True):
        Peaks.rechunk_on_save = do_rechunk

        with tempfile.TemporaryDirectory() as temp_dir:
            st = strax.Context(
                storage=strax.DataDirectory(path=temp_dir),
                register=[Records, Peaks],
                use_per_run_defaults=True,
            )

            peaks_0 = st.get_array("0", "peaks")
            peaks_0a = st.get_array("0", "peaks")
            np.testing.assert_array_equal(peaks_0, peaks_0a)


def test_run_selection():
    mock_rundb = [
        dict(name="0", mode="funny", tags=[dict(name="bad")]),
        dict(name="1", mode="nice", tags=[dict(name="interesting"), dict(name="bad")]),
        dict(name="2", mode="nice", tags=[dict(name="interesting")]),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        sf = strax.DataDirectory(path=temp_dir, deep_scan=True, provide_run_metadata=True)
        # Write mock runs db
        for d in mock_rundb:
            sf.write_run_metadata(d["name"], d)

        st = strax.Context(storage=sf)
        assert len(st.scan_runs()) == len(mock_rundb)
        assert st.run_metadata("0") == mock_rundb[0]
        assert st.run_metadata("0", projection="name") == {"name": "0"}

        assert len(st.select_runs(run_mode="nice")) == 2
        assert len(st.select_runs(include_tags="interesting")) == 2
        assert len(st.select_runs(include_tags="interesting", exclude_tags="bad")) == 1
        assert len(st.select_runs(include_tags="interesting", run_mode="nice")) == 2

        assert len(st.select_runs(run_id="0")) == 1
        assert len(st.select_runs(run_id="*", exclude_tags="bad")) == 1


def test_run_defaults():
    mock_rundb = [{"name": "0", strax.RUN_DEFAULTS_KEY: dict(base_area=43)}]

    with tempfile.TemporaryDirectory() as temp_dir:
        sf = strax.DataDirectory(path=temp_dir, deep_scan=True, provide_run_metadata=True)
        for d in mock_rundb:
            sf.write_run_metadata(d["name"], d)
        st = strax.Context(
            storage=sf,
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )

        # The run defaults get used
        peaks = st.get_array("0", "peaks")
        assert np.all(peaks["area"] == 43)

        # ... but the user can still override them
        peaks = st.get_array("0", "peaks", config=dict(base_area=44))
        assert np.all(peaks["area"] == 44)


def test_dtype_mismatch():
    mystrax = strax.Context(
        storage=[],
        register=[Records, Peaks],
        use_per_run_defaults=True,
        config=dict(give_wrong_dtype=True),
    )
    with pytest.raises(strax.PluginGaveWrongOutput):
        mystrax.get_array(run_id=run_id, targets="peaks")


def test_get_single_plugin():
    mystrax = strax.Context(
        storage=[],
        register=[Records, Peaks, PeakClassification],
        use_per_run_defaults=True,
    )
    p = mystrax.get_single_plugin("0", "peaks")
    p.empty_result()
    assert isinstance(p, Peaks)
    assert len(p.config)
    assert p.config["base_area"] == 0
    p = mystrax.get_single_plugin("0", "peak_classification")
    p.empty_result()


def test_allow_multiple(targets=("peaks", "records")):
    """Test if we can use the allow_multiple correctly and fail otherwise."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mystrax = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )
        mystrax.set_context_config({"allow_lazy": False, "timeout": 80})

        assert not mystrax.is_stored(run_id, "peaks")
        # Create everything at once with get_array and get_df should fail
        for function in [mystrax.get_array, mystrax.get_df]:
            try:
                function(run_id=run_id, allow_multiple=True, targets=targets)
            except RuntimeError:
                # Great, this doesn't work (and it shouldn't!)
                continue
            raise ValueError(f"{function} could run with allow_multiple")

        try:
            mystrax.make(run_id=run_id, targets=targets, processor="threaded_mailbox")
        except RuntimeError:
            # Great, we shouldn't be allowed
            pass

        assert not mystrax.is_stored(run_id, "peaks")
        mystrax.make(
            run_id=run_id, allow_multiple=True, targets=targets, processor="threaded_mailbox"
        )

        for t in targets:
            assert mystrax.is_stored(run_id, t)


def test_allow_multiple_inverted():
    # Make sure that the processing also works if the first target is
    # actually depending on the second. In that case, we should
    # subscribe the first target as the endpoint of the processing
    test_allow_multiple(
        targets=(
            "records",
            "peaks",
        )
    )


def test_available_for_run():
    """Very simply test the available_for_run function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mystrax = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )
        targets = list(mystrax._plugin_class_registry.keys())
        for exclude_i in range(len(targets)):
            for include_i in range(len(targets)):
                df = mystrax.available_for_run(
                    run_id, include_targets=targets[:include_i], exclude_targets=targets[:exclude_i]
                )
                if len(df):
                    # We haven't made any data
                    assert not sum(df["is_stored"])


def test_per_chunk_storage():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )

        # If per-chunk storage is not for dependencies, DataKey will not be different.
        key_1st = st.key_for(run_id, "records", chunk_number={"records": [0]})
        key_2nd = st.key_for(run_id, "records", chunk_number={"records": [1]})
        assert str(key_1st) == str(key_2nd)

        # If per-chunk storage is for dependencies, savers will not be different.
        components_1st = st.get_components(run_id, "peaks", chunk_number={"peaks": [0]})
        components_2nd = st.get_components(run_id, "peaks", chunk_number={"peaks": [1]})
        assert (
            components_1st.savers["peaks"][0].dirname == components_2nd.savers["peaks"][0].dirname
        )

        # Test merge_per_chunk_storage
        st.make(run_id, "records")
        n_chunks = len(st.get_metadata(run_id, "records")["chunks"])
        assert n_chunks > 2
        for i in range(n_chunks):
            st.make(run_id, "peaks", chunk_number={"records": [i]})
        assert not st.is_stored(run_id, "peaks")
        st.merge_per_chunk_storage(
            run_id, "peaks", "records", chunk_number_group=[[i] for i in range(n_chunks // 2)]
        )
        assert not st.is_stored(run_id, "peaks")
        st.merge_per_chunk_storage(run_id, "peaks", "records")
        assert st.is_stored(run_id, "peaks")
        with pytest.raises(ValueError):
            st.merge_per_chunk_storage(run_id, "peaks", "records")

        # Per-chunk storage not allowed for some plugins
        p = type("whatever", (strax.OverlapWindowPlugin,), dict(depends_on="records"))
        st.register(p)
        with pytest.raises(ValueError):
            st.make(run_id, "whatever", chunk_number={"records": [0]})


def test_dependency_tree():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(
            storage=strax.DataDirectory(temp_dir, deep_scan=True),
            register=[Records, Peaks],
            use_per_run_defaults=True,
        )
        st.tree
        st.inversed_tree
        st.tree_levels
