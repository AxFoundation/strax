from unittest import TestCase

import glob
import strax
from strax.testutils import Records
import os
import json
import tempfile
import numpy as np
import typing as ty
from immutabledict import immutabledict


class TestPerRunDefaults(TestCase):
    """Test the saving behavior of the context."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = self.tempdir.name
        self.st = strax.Context(
            use_per_run_defaults=True,
            register=[Records],
        )
        self.target = "records"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_write_data_dir(self):
        self.st.storage = [strax.DataDirectory(self.path)]
        run_id = "0"
        self.st.make(run_id, self.target)
        assert self.st.is_stored(run_id, self.target)

    def test_complain_run_id(self):
        self.st.storage = [strax.DataDirectory(self.path)]
        run_id = "run-0"
        with self.assertRaises(ValueError):
            self.st.make(run_id, self.target)


class VerboseDataDir(strax.DataDirectory):
    _verbose = False

    def _print(self, m):
        if self._verbose:
            print(m)

    def find(self, key, *args, **kwargs):
        message = f"{self.path} was asked for {key} ->"
        try:
            result = super().find(key, *args, **kwargs)
        except Exception as e:
            self._print(f"{message} raises {type(e)}")
            raise e
        self._print(f"{message} returns {result}")
        return result


class TestStorageType(TestCase):
    """Test that slow frontends are asked last to provide data.

    A bit of a clunky test to test a simple thing. It's may be a bit
    hidden but this is the goal:

     - Check that we are always asking for data to the *fastest* frontend first

    We do this by creating three frontends, of varying speed. To be able
    to be sure which frontend is returning what, we put a different
    amount of data in each of them, so we can be sure the right frontend
    is returning us that data (we do a lot of for loops where I don't
    otherwise now how we can cleanly extract which frontend is returning
    what).

    """

    target = "records"
    run_id = "1"

    context_kwargs = immutabledict(
        use_per_run_defaults=True,
        allow_rechunk=False,
        register=[Records],
    )
    _verbose = False

    @classmethod
    def setUpClass(cls) -> None:
        """Get a temp directory available of all the tests."""
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.path = cls.tempdir.name

    def tearDown(self):
        """After each test, delete the temporary directory."""
        self.tempdir.cleanup()

    def _sub_dir(self, subdir: str) -> str:
        return os.path.join(self.path, subdir)

    def get_st_and_fill_frontends(self) -> ty.Tuple[strax.Context, dict]:
        """Get options that allow us to do the check as in the docstring of this class."""
        # Three frontends, with three different names, remoteness levels
        # and number of chunks stored in them
        frontend_setup = {
            "name": ["far", "close", "intermediate"],
            "remoteness": [
                strax.StorageType.TAPE,
                strax.StorageType.LOCAL,
                strax.StorageType.REMOTE,
            ],
            "n_chunks": [1, 2, 3],
        }

        frontends = []
        for name, remoteness, n_chunks in zip(*list(frontend_setup.values())):
            sf = VerboseDataDir(self._sub_dir(name))
            sf.storage_type = remoteness
            temp_st = strax.Context(
                storage=[sf],
                config=dict(n_chunks=n_chunks),
                **self.context_kwargs,
            )
            # For each frontend, make data (that won't be equal size!)
            recs = temp_st.get_array(self.run_id, self.target)
            n_recs = len(recs)
            del recs
            print(f"{sf} ({name}) made {n_recs})")
            sf.readonly = True
            sf._verbose = self._verbose
            frontends += [sf]

        return (strax.Context(storage=frontends, **self.context_kwargs), frontend_setup)

    def test_dry_load_files(self):
        """Test that dry_load_files can load the data."""
        st, frontend_setup = self.get_st_and_fill_frontends()
        for sf in st.storage:
            key = st.key_for(self.run_id, self.target)
            dirname = os.path.join(sf.path, str(key))
            strax.io.dry_load_files(dirname)
            strax.io.dry_load_files(dirname, 0)
            with self.assertRaises(ValueError):
                strax.io.dry_load_files(dirname, 99)

    def test_close_goes_first_md(self):
        """Let's see that if we get the meta-data, it's from the one with the lowest remoteness.

        We can check this by comparing the number of chunks.

        """
        st, frontend_setup = self.get_st_and_fill_frontends()
        result = st.get_metadata(self.run_id, self.target)
        n_chunks = len(result["chunks"])
        closest = np.argmin(frontend_setup["remoteness"])
        n_fastest = frontend_setup["n_chunks"][closest]

        self.assertEqual(
            n_chunks,
            frontend_setup["n_chunks"][closest],
            f"Should have returned {n_fastest} from {st.storage[closest]}, got {n_chunks}",
        )

    def test_close_goes_first_on_loading(self):
        """Check that loading data comes from the fastest frontend by comparing the data length."""
        st, frontend_setup = self.get_st_and_fill_frontends()
        closest = np.argmin(frontend_setup["remoteness"])
        len_from_main_st = len(st.get_array(self.run_id, self.target))

        for sf_i, sf in enumerate(st.storage):
            st_compare = st.new_context()
            st_compare.storage = [sf]
            len_from_compare = len(st_compare.get_array(self.run_id, self.target))
            if sf_i == closest:
                self.assertEqual(len_from_compare, len_from_main_st)
            # else:
            #     self.assertNotEqual(len_from_compare, len_from_main_st)

    def test_check_chunk_n(self):
        """Check that StorageBackend detects when metadata is lying."""
        st, frontend_setup = self.get_st_and_fill_frontends()

        sf = st.storage[0]
        st_new = st.new_context()
        st_new.storage = [sf]
        key = st_new.key_for(self.run_id, self.target)
        backend, backend_key = sf.find(key, **st_new._find_options)
        prefix = strax.storage.files.dirname_to_prefix(backend_key)
        md = st_new.get_metadata(self.run_id, self.target)
        md["chunks"][0]["n"] += 1
        md_path = os.path.join(backend_key, f"{prefix}-metadata.json")
        with open(md_path, "w") as file:
            json.dump(md, file, indent=4)

        with self.assertRaises(strax.DataCorrupted):
            assert st_new.is_stored(self.run_id, self.target)
            st_new.get_array(self.run_id, self.target)

    def test_float_remoteness_allowed(self):
        """It can happen that the pre-defined remoteness identifiers in strax.StorageType are not
        sufficient, e.g. you have 10 similar but not quite the same performing frontends.

        You can set `sf.storage_type` as a float to fix this issue such
        that you can order an infinite amount of frontends (the intnum
        is only for readability).

        """

        # Two different classes with only slightly different storage_types
        class FrontentSlow(VerboseDataDir):
            storage_type = 10
            _verbose = True

        class FrontentSlightlySlower(VerboseDataDir):
            storage_type = 10.001
            raise_when_run = False
            _verbose = True

            def find(self, *args, **kwargs):
                print(self.raise_when_run)
                if self.raise_when_run:
                    raise strax.testutils.SomeCrash
                return super().find(*args, **kwargs)

        storage_slow = FrontentSlow(self._sub_dir("slow"))
        storage_slightly_slow = FrontentSlightlySlower(self._sub_dir("slightlyslow"))
        storages = [storage_slow, storage_slightly_slow]
        st = strax.Context(storage=storages, **self.context_kwargs)

        # Make the data, it should be in both frontends
        st.make(self.run_id, self.target)
        self.assertTrue(st.is_stored(self.run_id, self.target))
        for sf in storages:
            self.assertTrue(st._is_stored_in_sf(self.run_id, self.target, sf), str(sf))

        # Now set the charge, if the slightly slower frontend is asked
        # for data, it will raise an error
        storage_slightly_slow.raise_when_run = True
        st.set_context_config({"forbid_creation_of": "*"})
        # No error raises because we get the storage_slow's data
        st.get_array(self.run_id, self.target)
        # just to be sure, we would have gotten an error if it would
        # have gotten data from storage_slightly_slow
        with self.assertRaises(strax.testutils.SomeCrash):
            st.storage = [storage_slightly_slow]
            print(st.storage)
            st.is_stored(self.run_id, self.target)


class TestRechunking(TestCase):
    """Test the saving behavior of the context."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = self.tempdir.name
        self.st = strax.Context(
            use_per_run_defaults=True,
            register=[Records],
        )
        self.target = "records"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_rechunking(self):
        for compressor in strax.io.COMPRESSORS.keys():
            with self.subTest(compressor=compressor):
                self.setUp()
                self._rechunking(compressor)
                self.tearDown()

    def test_rechunk_parallelization(self):
        for parallel in [True, "process", False]:
            with self.subTest(parallel=parallel):
                self.setUp()
                self._rechunking(compressor="blosc", parallel=parallel)
                self.tearDown()

    def test_replace(self):
        self._rechunking(compressor="blosc", replace=True)

    def _rechunking(self, compressor, parallel=False, replace=False):
        """Test that we can use the strax.files.rechunking function to rechunk data outside the
        context."""
        target_path = tempfile.TemporaryDirectory()
        source_sf = strax.DataDirectory(self.path)
        st = self.st
        st.set_context_config(dict(allow_rechunk=False, n_chunks=10))
        st.storage = [source_sf]
        run_id = "0"
        st.make(run_id, self.target)
        assert st.is_stored(run_id, self.target)
        assert strax.utils.dir_size_mb(self.path) > 0
        original_n_files = len(glob.glob(os.path.join(self.path, "*", "*")))
        assert original_n_files > 3  # At least two files + metadata
        _, backend_key = source_sf.find(st.key_for(run_id, self.target))
        strax.rechunker(
            source_directory=backend_key,
            dest_directory=target_path.name if not replace else None,
            replace=True,
            compressor=compressor,
            target_size_mb=strax.default_chunk_size_mb * 2,
            parallel=parallel,
            max_workers=4,
            _timeout=5,
        )
        assert st.is_stored(run_id, self.target)
        # Should be empty, we just replaced the source
        assert strax.utils.dir_size_mb(target_path.name) == 0
        new_n_files = len(
            glob.glob(
                os.path.join(
                    self.path,
                    "*",
                    "*",
                )
            )
        )
        assert original_n_files > new_n_files
        st.set_context_config(dict(forbid_creation_of="*"))
        st.get_array(run_id, self.target)
        target_path.cleanup()


class TestBlosc(TestCase):
    """Blosc does not handle 2GB chunks, assert it fails with an useful error."""

    def test_blosc_fails(self):
        chunk = np.zeros(int(3e8), np.int64)
        assert chunk.nbytes > 2e9, "Test only works for >2GB chunks"
        assert chunk.nbytes < 3e9, "Don't go crazy here"
        with self.assertRaises(ValueError):
            strax.io.COMPRESSORS["blosc"]["compress"](chunk)
