import glob
import json
import os
import configparser
import os.path as osp
from typing import Optional
from bson import json_util
import shutil
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()

RUN_METADATA_PATTERN = "%s-metadata.json"
BUCKET_NAME = 'mlrice'


@export
class S3Frontend(StorageFrontend):
    """Simplest registry: single directory with FileStore data
    sitting in subdirectories.

    Run-level metadata is stored in loose json files in the directory.
    """

    can_define_runs = True
    provide_run_metadata = False
    provide_superruns = True
    BUCKET = "mlrice"

    def __init__(self, 
                 s3_access_key_id: str=None,
                 s3_secret_access_key: str=None,
                 endpoint_url=None,
                 path="", 
                 deep_scan=False, 
                 *args, 
                 **kwargs):
        """
        :param path: Path to folder with data subfolders.
        :param deep_scan: Let scan_runs scan over folders,
        so even data for which no run-level metadata is available
        is reported.

        For other arguments, see DataRegistry base class.
        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.deep_scan = deep_scan

        # Might need to reimplement this at some later time
        #if not self.readonly and not osp.exists(self.path):
        #    os.makedirs(self.path)

        self.config_path = os.getenv("XENON_CONFIG")
        if self.config_path is None:
            raise EnvironmentError("XENON_CONFIG file not found")
        else:
            self.config = configparser.ConfigParser()

            s3_access_key_id = self._get_config_value(s3_access_key_id, 
                                                      "s3_access_key_id")
            s3_secret_access_key = self._get_config_value(s3_secret_access_key, 
                                                          "s3_secret_access_key")
            endpoint_url = self._get_config_value(endpoint_url,
                                                  "endpoint_url")

        self.boto3_client_kwargs = {
            'aws_access_key_id': s3_access_key_id,
            'aws_secret_access_key': s3_secret_access_key,
            'endpoint_url': endpoint_url,
            'service_name': 's3',
            'config': Config(connect_timeout=5,
                             retries={'max_attempts': 10})}

        #  Initialized connection to S3-protocol storage
        self.s3 = boto3.client(**self.boto3_client_kwargs)

        self.backends = [S3Backend(**self.boto3_client_kwargs)]

    def _run_meta_path(self, run_id):
        return osp.join(self.path, RUN_METADATA_PATTERN % run_id)

    def run_metadata(self, run_id, projection=None):
        path = self._run_meta_path(run_id)
        # Changed ops. to self.s3 implementation
        if self.s3.list_objects_v2(Bucket=self.BUCKET,
                                   Prefix=path)['KeyCount'] == 0:
            raise strax.RunMetadataNotAvailable(
                f"No file at {path}, cannot find run metadata for {run_id}"
            )
        response = self.s3.get_object(Bucket=self.BUCKET, Key=path)
        metadata_content = response['Body'].read().decode('utf-8')
        md = json.loads(metadata_content, object_hook=json_util.object_hook)
        #with open(path, mode="r") as f:
        #    md = json.loads(f.read(), object_hook=json_util.object_hook)
        md = strax.flatten_run_metadata(md)
        if projection is not None:
            md = {k: v for k, v in md.items() if k in projection}
        return md

    def write_run_metadata(self, run_id, metadata):
        #response = self.s3.get_object(Bucket=self.BUCKET, Key=self._run_meta_path(run_id))
        #metadata_content = response['Body'].read().decode('utf-8')
        if "name" not in metadata:
            metadata["name"] = run_id
        
        self.s3.put_object(Bucket=self.BUCKET, 
                           Key=self._run_meta_path(run_id), 
                           Body=json.dumps(metadata, 
                                           sort_keys=True, 
                                           indent=4, 
                                           default=json_util.default))
        #with open(self._run_meta_path(run_id), mode="w") as f:
        #    if "name" not in metadata:
        #        metadata["name"] = run_id
        #    f.write(json.dumps(metadata, sort_keys=True, indent=4, default=json_util.default))

    def _scan_runs(self, store_fields):
        """Iterable of run document dictionaries.

        These should be directly convertable to a pandas DataFrame.

        """
        found = set()

        # Yield metadata for runs for which we actually have it
        for md_path in sorted(
            self.s3.list_objects_v2(Bucket = self.BUCKET,
                                    Prefix = osp.join(self.path, 
                                                      RUN_METADATA_PATTERN.replace("%s", "*")))
        ):
            # Parse the run metadata filename pattern.
            # (different from the folder pattern)
            run_id = osp.basename(md_path).split("-")[0]
            found.add(run_id)
            yield self.run_metadata(run_id, projection=store_fields)

        if self.deep_scan:
            # Yield runs for which no metadata exists
            # we'll make "metadata" that consist only of the run name
            for fn in self._subfolders():
                run_id = self._parse_folder_name(fn)[0]
                if run_id not in found:
                    found.add(run_id)
                    yield dict(name=run_id)

    def _find(self, key, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        self.raise_if_non_compatible_run_id(key.run_id)
        dirname = osp.join(self.path, str(key))
        exists = self.s3_object_exists(self.BUCKET, dirname)
        bk = self.backend_key(dirname)

        if write:
            if exists and not self._can_overwrite(key):
                raise strax.DataExistsError(at=dirname)
            return bk

        if allow_incomplete and not exists:
            # Check for incomplete data (only exact matching for now)
            if fuzzy_for or fuzzy_for_options:
                raise NotImplementedError(
                    "Mixing of fuzzy matching and allow_incomplete not supported by DataDirectory."
                )
            tempdirname = dirname + "_temp"
            bk = self.backend_key(tempdirname)
            if self.s3.list_objects_v2(Bucket=self.BUCKET,
                                       Prefix=tempdirname)['KeyCount'] >= 0:
                return bk

        # Check exact match
        if exists and self._folder_matches(dirname, key, None, None):
            return bk

        # Check metadata of all potentially matching data dirs for
        # matches. This only makes sense for fuzzy searches since
        # otherwise we should have had an exact match already. (Also
        # really slows down st.select runs otherwise because we doing an
        # entire search over all the files in self._subfolders for all
        # non-available keys).
        if fuzzy_for or fuzzy_for_options:
            for fn in self._subfolders():
                if self._folder_matches(fn, key, fuzzy_for, fuzzy_for_options):
                    return self.backend_key(fn)

        raise strax.DataNotAvailable

    def s3_object_exists(self, bucket_name, key):
        try:
            self.s3.head_object(Bucket=bucket_name, Key=key)
            return True  # Object exists
        except ClientError as e:
            # If a 404 error is returned, the object does not exist
            if e.response['Error']['Code'] == '404':
                return False
            else:
                # For any other error, you can choose to raise the exception or handle it
                raise e

    def _subfolders(self):
        """Loop over subfolders of self.path that match our folder format."""
        # Trigger if statement if path doesnt exist
        if self.s3.list_objects_v2(Bucket=self.BUCKET,
                                   Prefix=self.path)['KeyCount'] == 0:
            return
        for dirname in os.listdir(self.path):
            try:
                self._parse_folder_name(dirname)
            except InvalidFolderNameFormat:
                continue
            yield osp.join(self.path, dirname)

    @staticmethod
    def _parse_folder_name(fn):
        """Return (run_id, data_type, hash) if folder name matches DataDirectory convention, raise
        InvalidFolderNameFormat otherwise."""
        stuff = osp.normpath(fn).split(os.sep)[-1].split("-")
        if len(stuff) != 3:
            # This is not a folder with strax data
            raise InvalidFolderNameFormat(fn)
        return stuff

    def _folder_matches(self, fn, key, fuzzy_for, fuzzy_for_options, ignore_name=False):
        """Return the run_id of folder fn if it matches key, or False if it does not.

        :param name: Ignore the run name part of the key. Useful for listing availability.

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
        return self.backends[0].__class__.__name__, dirname

    def remove(self, key):
        # There is no database, so removing the folder from the filesystem
        # (which FileStore should do) is sufficient.
        pass

    @staticmethod
    def raise_if_non_compatible_run_id(run_id):
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
    """Store data locally in a directory of binary files.

    Files are named after the chunk number (without extension). Metadata is stored in a file called
    metadata.json.

    """

    BUCKET = 'mlrice'

    def __init__(
        self,
        *args,
        set_target_chunk_mb: Optional[int] = None,
        **kwargs,
    ):
        """Add set_chunk_size_mb to strax.StorageBackend to allow changing the chunk.target_size_mb
        returned from the loader, any args or kwargs are passed to the strax.StorageBackend.

        :param set_target_chunk_mb: Prior to returning the loaders' chunks, return the chunk with an
            updated target size

        """
        super().__init__()
        self.s3 = boto3.client(**kwargs)
        self.set_chunk_size_mb = set_target_chunk_mb

    def _get_metadata(self, dirname):
        prefix = dirname_to_prefix(dirname)
        metadata_json = f"{prefix}-metadata.json"
        md_path = osp.join(dirname, metadata_json)

        if self.s3.list_objects_v2(Bucket=self.BUCKET,
                                   Prefix=md_path)['KeyCount'] == 0:
            # Try to see if we are so fast that there exists a temp folder
            # with the metadata we need.
            md_path = osp.join(dirname + "_temp", metadata_json)

        if self.s3.list_objects_v2(Bucket = self.BUCKET,
                                   Prefix=md_path)['KeyCount'] == 0:
            # Try old-format metadata
            # (if it's not there, just let it raise FileNotFound
            # with the usual message in the next stage)
            old_md_path = osp.join(dirname, "metadata.json")
            if self.s3.list_objects_v2(Bucket=self.BUCKET,
                                       Prefix=old_md_path)['KeyCount'] == 0:
                raise strax.DataCorrupted(f"Data in {dirname} has no metadata")
            md_path = old_md_path

        
        response = self.s3.get_object(Bucket=self.BUCKET, Key=md_path)
        metadata_content = response['Body'].read().decode('utf-8')
        return json.loads(metadata_content)
        #with open(md_path, mode="r") as f:
        #    return json.loads(f.read())

    def _read_and_format_chunk(self, *args, **kwargs):
        chunk = super()._read_and_format_chunk(*args, **kwargs)
        if self.set_chunk_size_mb:
            chunk.target_size_mb = self.set_chunk_size_mb
        return chunk

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        fn = osp.join(dirname, chunk_info["filename"])
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, **kwargs):
        # Test if the parent directory is writeable.
        # We need abspath since the dir itself may not exist,
        # even though its parent-to-be does
        parent_dir = os.path.abspath(os.path.join(dirname, os.pardir)) # This might need some work

        # In case the parent dir also doesn't exist, we have to create is
        # otherwise the write permission check below will certainly fail
        # I dont think this is needed for S3 so we can delete it
        #try:
        #    os.makedirs(parent_dir, exist_ok=True)
        #except OSError as e:
        #    raise strax.DataNotAvailable(
        #        f"Can't write data to {dirname}, "
        #        f"{parent_dir} does not exist and we could not create it."
        #        f"Original error: {e}"
        #    )

        # Finally, check if we have permission to create the new subdirectory
        # (which the Saver will do)
        # Also dont think its needed
        #if not os.access(parent_dir, os.W_OK):
        #    raise strax.DataNotAvailable(
        #        f"Can't write data to {dirname}, no write permissions in {parent_dir}."
        #    )

        return S3Saver(dirname, metadata=metadata, **kwargs)


@export
class S3Saver(strax.Saver):
    """Saves data to compressed binary files."""

    json_options = dict(sort_keys=True, indent=4)
    # When writing chunks, rewrite the json file every time we write a chunk
    _flush_md_for_every_chunk = True

    def __init__(self, dirname, metadata, **kwargs):
        super().__init__(metadata = metadata)
        self.dirname = dirname
        self.tempdirname = dirname + "_temp"
        self.prefix = dirname_to_prefix(dirname)
        self.metadata_json = f"{self.prefix}-metadata.json"
        self.s3 = boto3.client(**kwargs)
        self.bucket_name = "mlrice"

        self.config = boto3.s3.transfer.TransferConfig(
            max_concurrency=40,
            num_download_attempts=30)

        if self.s3.list_objects_v2(Bucket=BUCKET_NAME,
                                       Prefix=dirname)['KeyCount'] == 1:
            print(f"Removing data in {dirname} to overwrite")
            shutil.rmtree(dirname)
        if self.s3.list_objects_v2(Bucket=BUCKET_NAME,
                                       Prefix=self.tempdirname)['KeyCount'] == 0:
            print(f"Removing old incomplete data in {self.tempdirname}")
            shutil.rmtree(self.tempdirname)
        #os.makedirs(self.tempdirname)
        self._flush_metadata()

    def _flush_metadata(self):
        # Convert the metadata dictionary to a JSON string
        metadata_content = json.dumps(self.md, **self.json_options)
        
        # Define the S3 key for the metadata file (similar to a file path in a traditional file system)
        metadata_key = f"{self.tempdirname}/{self.metadata_json}"
    
        # Upload the metadata to S3
        self.s3.put_object(Bucket=self.bucket_name, Key=metadata_key, Body=metadata_content)

    def _chunk_filename(self, chunk_info):
        if "filename" in chunk_info:
            return chunk_info["filename"]
        ichunk = "%06d" % chunk_info["chunk_i"]
        return f"{self.prefix}-{ichunk}"

    def _save_chunk(self, data, chunk_info, executor=None):
        filename = self._chunk_filename(chunk_info)

        fn = os.path.join(self.tempdirname, filename)
        kwargs = dict(data=data, compressor=self.md["compressor"])
        if executor is None:
            filesize = strax.save_file(fn, **kwargs)
            fn.seek(0)
            # Giving just the filename wont work, needs to be the full path
            self.s3.upload_fileobj(fn,
                                   BUCKET_NAME,
                                   filename,
                                   Config=self.config,)
            return dict(filename=filename, filesize=filesize), None
        else:
            # Might need to add some s3 stuff here
            return dict(filename=filename), executor.submit(strax.save_file, fn, **kwargs)

    def _save_chunk_metadata(self, chunk_info):
        is_first = chunk_info["chunk_i"] == 0
        if is_first:
            self.md["start"] = chunk_info["start"]

        if self.is_forked:
            # Do not write to the main metadata file to avoid race conditions
            # Instead, write a separate metadata.json file for this chunk,
            # to be collected later.

            # We might not have a filename yet:
            # the chunk is not saved when it is empty
            filename = self._chunk_filename(chunk_info)

            fn = f"{self.tempdirname}/metadata_{filename}.json"
            metadata_content = json.dump(chunk_info, **self.json_options)
            self.s3.put_object(Bucket=self.bucket_name, Key=fn, Body=metadata_content)
            #with open(fn, mode="w") as f:
            #    f.write(json.dumps(chunk_info, **self.json_options))

        # To ensure we have some metadata to load with allow_incomplete,
        # modify the metadata immediately for the first chunk.
        # If we are forked, modifying self.md is harmless since
        # we're in a different process.

        if not self.is_forked or is_first:
            # Just append and flush the metadata
            # (maybe not super-efficient to write the json every time...
            # just don't use thousands of chunks)
            self.md["chunks"].append(chunk_info)
            if self._flush_md_for_every_chunk:
                self._flush_metadata()

    def _close(self):
        # Check if temp directory exists in the S3 bucket by listing objects with the tempdirname prefix
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.tempdirname)
            if 'Contents' not in response or len(response['Contents']) == 0:
                raise RuntimeError(
                    f"{self.tempdirname} was already renamed to {self.dirname}. "
                    "Did you attempt to run two savers pointing to the same "
                    "directory? Otherwise, this could be a strange race "
                    "condition or bug."
                )

            # List the files in the temporary directory matching metadata_*.json
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=f"{self.tempdirname}/metadata_")
            for obj in response.get('Contents', []):
                key = obj['Key']
                # Download each metadata file, process, and delete from tempdirname
                metadata_object = self.s3.get_object(Bucket=self.bucket_name, Key=key)
                metadata_content = metadata_object['Body'].read().decode('utf-8')
                self.md["chunks"].append(json.loads(metadata_content))

                # Optionally, delete the metadata file after processing
                self.s3.delete_object(Bucket=self.bucket_name, Key=key)

            # Flush metadata (this would be another method to handle your metadata saving logic)
            self._flush_metadata()

            # Rename the directory by copying all files from tempdirname to dirname and deleting from tempdirname
            self._rename_s3_folder(self.tempdirname, self.dirname)

        except ClientError as e:
            print(f"Error occurred: {e}")
            raise

    def _rename_s3_folder(self, tempdirname, dirname):
        # List the files in the temporary directory
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=tempdirname)
        for obj in response.get('Contents', []):
            key = obj['Key']
            # Copy each file from the temporary directory to the final directory
            new_key = key.replace(tempdirname, dirname)
            self.s3.copy_object(Bucket=self.bucket_name, CopySource={'Bucket': self.bucket_name, 'Key': key}, Key=new_key)
            # Delete the file from the temporary directory
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)

        # Delete the temporary directory
        self.s3.delete_object(Bucket=self.bucket_name, Key=tempdirname)


@export
class InvalidFolderNameFormat(Exception):
    pass