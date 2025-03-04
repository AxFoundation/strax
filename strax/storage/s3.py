import json
import os
import os.path as osp
from typing import Optional
from bson import json_util
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()

RUN_METADATA_PATTERN = "%s-metadata.json"
BUCKET_NAME = "mlrice"


@export
class S3Frontend(StorageFrontend):
    """A storage frontend that interacts with an S3-compatible object storage.

    This class handles run-level metadata storage and retrieval, as well as scanning for available
    runs.

    """

    can_define_runs = True
    provide_run_metadata = False
    provide_superruns = True
    BUCKET = "mlrice"

    def __init__(
        self,
        s3_access_key_id: str = "",
        s3_secret_access_key: str = "",
        endpoint_url: str = "https://rice1.osn.mghpcc.org/",
        path: str = "",
        bucket_name: str = "",
        deep_scan: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize S3Frontend with given storage parameters.

        :param s3_access_key_id: AWS access key for authentication.
        :param s3_secret_access_key: AWS secret key for authentication.
        :param endpoint_url: URL of the S3-compatible object storage.
        :param path: Base path for storing data.
        :param bucket_name: Name of the S3 bucket to use.
        :param deep_scan: If True, scans for runs even without explicit metadata.
        :param args: Additional arguments passed to the superclass.
        :param kwargs: Additional keyword arguments passed to the superclass. For other arguments,
            see DataRegistry base class.

        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.deep_scan = deep_scan

        # Configure S3 client
        self.boto3_client_kwargs = {
            "aws_access_key_id": s3_access_key_id,
            "aws_secret_access_key": s3_secret_access_key,
            "endpoint_url": endpoint_url,
            "service_name": "s3",
            "config": Config(connect_timeout=5, retries={"max_attempts": 10}),
        }

        if bucket_name != "":
            self.bucket_name = bucket_name

        #  Initialized connection to S3 storage
        self.s3 = boto3.client(**self.boto3_client_kwargs)
        self.backends = [S3Backend(self.bucket_name, **self.boto3_client_kwargs)]

        if s3_access_key_id != "":
            self.is_configed = True
        else:
            self.is_configed = False

    def _run_meta_path(self, run_id: str) -> str:
        """Generate the metadata file path for a given run ID.

        :param run_id: The identifier of the run.
        :return: The path where the metadata is stored.

        """
        return osp.join(self.path, RUN_METADATA_PATTERN % run_id)

    def run_metadata(self, run_id: str, projection=None) -> dict:
        """Retrieve metadata for a given run from S3.

        Parameters
        ----------

        run_id : (str)
            The identifier of the run.
        projection :
            Fields to extract from metadata (optional).
        :return: Run metadata as a dictionary.

        """
        path = self._run_meta_path(run_id)

        # Checks if metadata exists
        if self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=path)["KeyCount"] == 0:
            raise strax.RunMetadataNotAvailable(
                f"No file at {path}, cannot find run metadata for {run_id}"
            )

        # Retrieve metadata
        response = self.s3.get_object(Bucket=self.bucket_name, Key=path)
        metadata_content = response["Body"].read().decode("utf-8")
        md = json.loads(metadata_content, object_hook=json_util.object_hook)
        md = strax.flatten_run_metadata(md)

        if projection is not None:
            md = {key: value for key, value in md.items() if key in projection}
        return md

    def write_run_metadata(self, run_id: str, metadata: dict):
        """Write metadata for a specific run to S3.

        :param run_id: The identifier of the run.
        :param metadata: The metadata dictionary to store.

        """
        if "name" not in metadata:
            metadata["name"] = run_id

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=self._run_meta_path(run_id),
            Body=json.dumps(metadata, sort_keys=True, indent=4, default=json_util.default),
        )

    def s3_object_exists(self, key) -> bool:
        """Check if a given object exists in the S3 bucket.

        :param key: The object key to check.
        :return: True if the object exists, otherwise False.

        """
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=key, Delimiter="/")

            # Check if any objects were returned, and if the key exactly matches
            if "CommonPrefixes" in response:
                for prefix in response["CommonPrefixes"]:
                    new_key = key + "/"
                    if prefix["Prefix"].startswith(new_key):
                        return True  # Object exists

            return False  # Object does not exist
        except ClientError as e:
            raise e

    def _scan_runs(self, store_fields):
        """Scan for available runs stored in S3.

        :param store_fields: List of metadata fields to return.
        :return: Yields dictionaries of run metadata.

        """
        found = set()

        # Retrieve stored runs from S3
        for md_path in sorted(
            self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=osp.join(self.path, RUN_METADATA_PATTERN.replace("%s", "*")),
            )
        ):
            run_id = osp.basename(md_path).split("-")[0]
            found.add(run_id)
            yield self.run_metadata(run_id, projection=store_fields)

        # Preform deepscan if enabled
        if self.deep_scan:
            for fn in self._subfolders():
                run_id = self._parse_folder_name(fn)[0]
                if run_id not in found:
                    found.add(run_id)
                    yield dict(name=run_id)

    def _find(self, key, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        """Find the appropriate storage key for a given dataset.

        :param key: The dataset key.
        :param write: Whether to check for writable access.
        :param allow_incomplete: Allow incomplete datasets.
        :param fuzzy_for: Parameters for fuzzy search.
        :param fuzzy_for_options: Additional fuzzy search options.
        :return: The backend key if found, otherwise raises DataNotAvailable.

        """

        self.raise_if_non_compatible_run_id(key.run_id)
        dirname = osp.join(self.path, str(key))
        exists = self.s3_object_exists(self.bucket_name, dirname)
        bk = self.backend_key(dirname)

        if write:
            if exists and not self._can_overwrite(key):
                raise strax.DataExistsError(at=dirname)
            return bk

        if allow_incomplete and not exists:
            # Check for incomplete data (only exact matching for now)
            if fuzzy_for or fuzzy_for_options:
                raise NotImplementedError(
                    "Mixing of fuzzy matching and allow_incomplete not",
                    " supported by DataDirectory.",
                )
            tempdirname = dirname + "_temp"
            bk = self.backend_key(tempdirname)
            if (
                self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=tempdirname)["KeyCount"]
                >= 0
            ):
                return bk

        # Check exact match
        if exists and self._folder_matches(dirname, key, None, None):
            return bk

        # If fuzzy search is enabled find fuzzy file names
        if fuzzy_for or fuzzy_for_options:
            for fn in self._subfolders():
                if self._folder_matches(fn, key, fuzzy_for, fuzzy_for_options):
                    return self.backend_key(fn)

        raise strax.DataNotAvailable

    def _get_config_value(self, variable, option_name):
        """Retrieve a configuration value from the environment or config file.

        :param variable: The variable to check.
        :param option_name: The option name in the config file.
        :return: The retrieved configuration value.

        """
        if variable is None:
            if "s3" not in self.config:
                raise EnvironmentError("S3 access point not spesified")
            if not self.config.has_option("s3", option_name):
                raise EnvironmentError(f"S3 access point lacks a {option_name}")
            return self.config.get("s3", option_name)

        else:
            return variable

    def _subfolders(self):
        """Loop over subfolders of self.path that match our folder format."""
        # Trigger if statement if path doesnt exist
        if self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.path)["KeyCount"] == 0:
            return
        for dirname in os.listdir(self.path):
            try:
                self._parse_folder_name(dirname)
            except InvalidFolderNameFormat:
                continue
            yield osp.join(self.path, dirname)

    def _folder_matches(self, fn, key, fuzzy_for, fuzzy_for_options, ignore_name=False):
        """Check if a folder matches the required data key.

        :param fn: Folder name.
        :param key: Data key to match against.
        :param fuzzy_for: Parameters for fuzzy search.
        :param fuzzy_for_options: Additional fuzzy search options.
        :param ignore_name: If True, ignores run name while matching.
        :return: The run_id if it matches, otherwise False.

        """
        # Parse the folder name
        try:
            _run_id, _data_type, _hash = self._parse_folder_name(fn)
        except InvalidFolderNameFormat:
            return False

        # Check exact match
        if _data_type != key.data_type:
            return False
        if not ignore_name and _run_id != key._run_id:
            return False

        # Check fuzzy match
        if not (fuzzy_for or fuzzy_for_options):
            if _hash == key.lineage_hash:
                return _run_id
            return False
        metadata = self.backends[0].get_metadata(fn)
        if self._matches(metadata["lineage"], key.lineage, fuzzy_for, fuzzy_for_options):
            return _run_id
        return False

    def backend_key(self, dirname):
        """Return the backend key representation.

        :param dirname: The directory name.
        :return: Backend key tuple.

        """
        return self.backends[0].__class__.__name__, dirname

    def remove(self, key):
        # Remove a data entery from storage
        NotImplementedError

    @staticmethod
    def _parse_folder_name(fn):
        """Return (run_id, data_type, hash) if folder name matches DataDirectory convention, raise
        InvalidFolderNameFormat otherwise."""
        stuff = osp.normpath(fn).split(os.sep)[-1].split("-")
        if len(stuff) != 3:
            # This is not a folder with strax data
            raise InvalidFolderNameFormat(fn)
        return stuff

    @staticmethod
    def raise_if_non_compatible_run_id(run_id):
        """Raise an error if the run ID contains invalid characters.

        :param run_id: The run identifier.

        """
        if "-" in str(run_id):
            raise ValueError(
                "The filesystem frontend does not understand"
                " run_id's with '-', please replace with '_'"
            )


@export
def dirname_to_prefix(dirname):
    """Return filename prefix from dirname."""
    dirname = dirname.replace("_temp", "")
    return os.path.basename(dirname.strip("/").rstrip("\\")).split("-", maxsplit=1)[1]


@export
class S3Backend(strax.StorageBackend):
    """A storage backend that stores data in an S3-compatible object storage.

    Data is stored in binary files, named based on the chunk number. Metadata is stored separately
    as a JSON file.

    """

    BUCKET = "mlrice"

    def __init__(
        self,
        bucket_name,
        set_target_chunk_mb: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Add set_chunk_size_mb to strax.StorageBackend to allow changing the chunk.target_size_mb
        returned from the loader, any args or kwargs are passed to the strax.StorageBackend.

        :param bucket_name: Name of the bucket used as storage
        :param set_target_chunk_mb: Prior to returning the loaders' chunks, return the chunk with an
            updated target size

        """
        super().__init__()
        self.s3 = boto3.client(**kwargs)
        self.set_chunk_size_mb = set_target_chunk_mb
        self.bucket_name = bucket_name

    def _get_metadata(self, dirname):
        """Retrieve metadata for a given directory in S3.

        :param dirname: The directory name in S3 where metadata is stored.
        :return: Dictionary containing metadata information.
        :raises strax.DataCorrupted: If metadata is missing or corrupted.

        """
        prefix = dirname_to_prefix(dirname)
        metadata_json = f"{prefix}-metadata.json"
        md_path = osp.join(dirname, metadata_json)

        if self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=md_path)["KeyCount"] == 0:
            # Try to see if we are so fast that there exists a temp folder
            # with the metadata we need.
            md_path = osp.join(dirname + "_temp", metadata_json)

        if self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=md_path)["KeyCount"] == 0:
            # Try old-format metadata
            # (if it's not there, just let it raise FileNotFound
            # with the usual message in the next stage)
            old_md_path = osp.join(dirname, "metadata.json")
            if (
                self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=old_md_path)["KeyCount"]
                == 0
            ):
                raise strax.DataCorrupted(f"Data in {dirname} has no metadata")
            md_path = old_md_path

        response = self.s3.get_object(Bucket=self.bucket_name, Key=md_path)
        metadata_content = response["Body"].read().decode("utf-8")
        return json.loads(metadata_content)
        # with open(md_path, mode="r") as f:
        #    return json.loads(f.read())

    def _read_and_format_chunk(self, *args, **kwargs):
        """Read a data chunk and optionally update its target size.

        :return: Formatted data chunk.

        """
        chunk = super()._read_and_format_chunk(*args, **kwargs)
        if self.set_chunk_size_mb:
            chunk.target_size_mb = self.set_chunk_size_mb
        return chunk

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        """Read a chunk of data from S3.

        :param dirname: Directory in S3 containing the chunk.
        :param chunk_info: Dictionary containing chunk metadata.
        :param dtype: Data type of the chunk.
        :param compressor: Compression format used for the chunk.
        :return: Loaded data chunk.

        """
        fn = osp.join(dirname, chunk_info["filename"])
        return strax.load_file(
            fn,
            dtype=dtype,
            compressor=compressor,
            S3_client=self.s3,
            bucket_name=self.bucket_name,
            is_s3_path=True,
        )

    def _saver(self, dirname, metadata, **kwargs):
        """Create a saver object for writing data to S3.

        :param dirname: Directory in S3 where data will be stored.
        :param metadata: Metadata dictionary associated with the data.
        :param kwargs: Additional keyword arguments for the saver.
        :return: An instance of `S3Saver`.

        """

        parent_dir = os.path.abspath(os.path.join(dirname, os.pardir))

        return S3Saver(parent_dir, self.s3, self.bucket_name, metadata=metadata, **kwargs)


@export
class S3Saver(strax.Saver):
    """A saver class that writes data chunks to an S3-compatible storage backend.

    Supports metadata management and chunked data saving.

    """

    json_options = dict(sort_keys=True, indent=4)
    # When writing chunks, rewrite the json file every time we write a chunk
    _flush_md_for_every_chunk = True

    def __init__(self, dirname, s3, bucket_name, metadata, **kwargs):
        """Initialize the S3Saver instance.

        :param dirname: Directory path (prefix) in S3 where data is stored.
        :param s3: Boto3 S3 client instance.
        :param metadata: Metadata dictionary associated with the data.
        :param kwargs: Additional keyword arguments for the saver.

        """
        super().__init__(metadata=metadata)
        self.dirname = dirname
        self.s3 = s3
        self.bucket_name = bucket_name

        self.tempdirname = dirname + "_temp"
        self.prefix = dirname_to_prefix(dirname)
        self.metadata_json = f"{self.prefix}-metadata.json"

        self.config = boto3.s3.transfer.TransferConfig(max_concurrency=40, num_download_attempts=30)

        if self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=dirname)["KeyCount"] == 1:
            print(f"Removing data in {dirname} to overwrite")
            self.s3.delete_object(Bucket=self.bucket_name, Key=dirname)
        if (
            self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.tempdirname)["KeyCount"]
            == 1
        ):
            print(f"Removing old incomplete data in {self.tempdirname}")
            self.s3.delete_object(Bucket=self.bucket_name, Key=dirname)
        # os.makedirs(self.tempdirname)
        self._flush_metadata()

    def _flush_metadata(self):
        # Convert the metadata dictionary to a JSON string
        metadata_content = json.dumps(self.md, **self.json_options)

        # Define the S3 key for the metadata file
        metadata_key = f"{self.tempdirname}/{self.metadata_json}"

        # Upload the metadata to S3
        self.s3.put_object(Bucket=self.bucket_name, Key=metadata_key, Body=metadata_content)

    def _chunk_filename(self, chunk_info):
        """Generate a filename for a given chunk.

        :param chunk_info: Dictionary containing chunk metadata.
        :return: Filename string.

        """
        if "filename" in chunk_info:
            return chunk_info["filename"]
        ichunk = "%06d" % chunk_info["chunk_i"]
        return f"{self.prefix}-{ichunk}"

    def _save_chunk(self, data, chunk_info, executor=None):
        """Save a chunk of data to S3.

        :param data: Data chunk to be saved.
        :param chunk_info: Metadata dictionary for the chunk.
        :param executor: Optional executor for parallel writes.
        :return: Chunk metadata dictionary.

        """
        filename = self._chunk_filename(chunk_info)

        fn = os.path.join(self.tempdirname, filename)
        kwargs = dict(data=data, compressor=self.md["compressor"])
        if executor is None:
            filesize = strax.save_file(fn, is_s3_path=True, **kwargs)
            return dict(filename=filename, filesize=filesize), None
        else:
            # Might need to add some s3 stuff here
            return dict(filename=filename), executor.submit(
                strax.save_file, fn, is_s3_path=True, **kwargs
            )

    def _save_chunk_metadata(self, chunk_info):
        """Save metadata associated with a data chunk.

        :param chunk_info: Dictionary containing chunk metadata.

        """
        is_first = chunk_info["chunk_i"] == 0
        if is_first:
            self.md["start"] = chunk_info["start"]

        if self.is_forked:
            # Do not write to the main metadata file to avoid race conditions

            filename = self._chunk_filename(chunk_info)
            fn = f"{self.tempdirname}/metadata_{filename}.json"
            metadata_content = json.dump(chunk_info, **self.json_options)
            self.s3.put_object(Bucket=self.s3.bucket_name, Key=fn, Body=metadata_content)

        if not self.is_forked or is_first:
            self.md["chunks"].append(chunk_info)
            if self._flush_md_for_every_chunk:
                self._flush_metadata()

    def _close(self):
        """Finalize the saving process by merging temp data and flushing metadata."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.s3.bucket_name, Prefix=self.tempdirname)
            if "Contents" not in response or len(response["Contents"]) == 0:
                raise RuntimeError(
                    f"{self.tempdirname} was already renamed to {self.dirname}. "
                    "Did you attempt to run two savers pointing to the same "
                    "directory? Otherwise, this could be a strange race "
                    "condition or bug."
                )

            # List the files in the temporary directory matching metadata_*.json
            response = self.s3.list_objects_v2(
                Bucket=self.s3.bucket_name, Prefix=f"{self.tempdirname}/metadata_"
            )
            for obj in response.get("Contents", []):
                key = obj["Key"]
                # Download each metadata file, process, and delete from tempdirname
                metadata_object = self.s3.get_object(Bucket=self.s3.bucket_name, Key=key)
                metadata_content = metadata_object["Body"].read().decode("utf-8")
                self.md["chunks"].append(json.loads(metadata_content))

                # Optionally, delete the metadata file after processing
                self.s3.delete_object(Bucket=self.s3.bucket_name, Key=key)

            # Flush metadata (this would be another method to handle your metadata saving logic)
            self._flush_metadata()

            # Rename directory by copying all files from tempdirname to dirname
            self._rename_s3_folder(self.tempdirname, self.dirname)

        except ClientError as e:
            print(f"Error occurred: {e}")
            raise

    def _rename_s3_folder(self, tempdirname, dirname):
        """Rename the temporary directory to the final storage location in S3.

        :param tempdirname: Temporary directory path in S3.
        :param dirname: Final directory path in S3.

        """
        response = self.s3.list_objects_v2(Bucket=self.s3.bucket_name, Prefix=tempdirname)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            # Copy each file from the temporary directory to the final directory
            new_key = key.replace(tempdirname, dirname)
            self.s3.copy_object(
                Bucket=self.s3.bucket_name,
                CopySource={"Bucket": self.s3.bucket_name, "Key": key},
                Key=new_key,
            )
            # Delete the file from the temporary directory
            self.s3.delete_object(Bucket=self.s3.bucket_name, Key=key)

        # Delete the temporary directory
        self.s3.delete_object(Bucket=self.s3.bucket_name, Key=tempdirname)


@export
class InvalidFolderNameFormat(Exception):
    pass
