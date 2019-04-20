"""I/O that speaks the S3 protocol

The S3 protocol is an HTTP-based protocol spoken by Amazon Web Services, but
also storage systems such as Ceph (used in particle physics).  Therefore,
this can be widely used if you know the appropriate endpoint.

Be aware that you must specify the following two environmental variables or
pass the appropriate valeus to the constructor:

  *  S3_ACCESS_KEY_ID
  *  S3_SECRET_ACCESS_KEY

"""

import json
import os
import tempfile

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

import strax
from strax import StorageFrontend

export, __all__ = strax.exporter()

# Track versions of S3 interface
VERSION = 2
BUCKET_NAME = 'snax_s3_v%d' % VERSION


@export
class SimpleS3Store(StorageFrontend):
    """Frontend for S3 stores that just checks if data exists.

    This uses boto3 for communicating, where you can look at their docs
    to understand a lot of this.  S3 is an object store where each object is
    chunk.  The bucket corresponds to strax key (run / plugin).

    Currently, no run level metadata is stored.
    """

    def __init__(self,
                 s3_access_key_id=None,
                 s3_secret_access_key=None,
                 endpoint_url='http://ceph-s3.mwt2.org',
                 *args, **kwargs):
        """
        You must provide credentials to access your storage element.

        :param s3_access_key_id: access key for S3-readable storage.
        :param s3_secret_access_key: secret key for S3-readable storage.
        :param endpoint_url: URL of S3-readable storage.

        For other arguments, see DataRegistry base class.
        """

        super().__init__(*args, **kwargs)

        # Get S3 protocol credentials
        if s3_access_key_id is None:
            if 'S3_ACCESS_KEY_ID' not in os.environ:
                raise EnvironmentError("S3 access key not specified")
            s3_access_key_id = os.environ.get('S3_ACCESS_KEY_ID')
        if s3_secret_access_key is None:
            if 'S3_SECRET_ACCESS_KEY' not in os.environ:
                raise EnvironmentError("S3 secret key not specified")
            s3_secret_access_key = os.environ.get('S3_SECRET_ACCESS_KEY')

        self.boto3_client_kwargs = {
            'aws_access_key_id': s3_access_key_id,
            'aws_secret_access_key': s3_secret_access_key,
            'endpoint_url': endpoint_url,
            'service_name': 's3',
            'config': Config(connect_timeout=5,
                             retries={'max_attempts': 10})}

        #  Initialized connection to S3-protocol storage
        self.s3 = boto3.client(**self.boto3_client_kwargs)

        # Create bucket (does nothing if exists)
        # self.s3.create_bucket(Bucket=BUCKET_NAME)

        # Setup backends for reading
        self.backends = [S3Backend(**self.boto3_client_kwargs), ]

    def _find(self, key, write,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        """Determine if data exists

        Search the S3 store to see if data is there.
        """
        if fuzzy_for or fuzzy_for_options:
            raise NotImplementedError("Can't do fuzzy with S3")

        key_str = str(key)
        bk = self.backend_key(key_str)

        try:
            self.backends[0].get_metadata(key)
        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                if write:
                    return bk
                else:
                    raise strax.DataNotAvailable
            else:
                raise ex

        if write and not self._can_overwrite(key):
            raise strax.DataExistsError(at=bk)
        return bk

    def backend_key(self, key_str):

        return self.backends[0].__class__.__name__, key_str

    def remove(self, key):
        raise NotImplementedError()


@export
class S3Backend(strax.StorageBackend):
    """Store data in S3-backend

    Buckets are run/plugin keys and objects are chunks.
    """

    def __init__(self, **kwargs):
        super().__init__()

        #  Initialized connection to S3-protocol storage
        self.s3 = boto3.client(**kwargs)
        self.kwargs = kwargs  # Used later for setting up Saver

    def get_metadata(self, backend_key):
        # Grab metadata object from S3 bucket
        result = self.s3.get_object(Bucket=BUCKET_NAME,
                                    Key=f'{backend_key}/metadata.json')
        # Then read/parse it into ASCII
        text = result["Body"].read().decode()

        # Before returning dictionary
        return json.loads(text)

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        # Temporary hack for backward compatibility
        if 'filename' in chunk_info:
            chunk_info['key_name'] = f"{backend_key}/{chunk_info['filename']}"

        with tempfile.SpooledTemporaryFile() as f:
            self.s3.download_fileobj(Bucket=BUCKET_NAME,
                                     Key=chunk_info['key_name'],
                                     Fileobj=f)
            f.seek(0)  # Needed?
            return strax.load_file(f,
                                   dtype=dtype,
                                   compressor=compressor)

    def _saver(self, key, metadata):
        return S3Saver(key,
                       metadata=metadata,
                       **self.kwargs)


@export
class S3Saver(strax.Saver):
    """Saves data to S3-compatible storage
    """
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, key, metadata=None, **kwargs):
        if metadata is None:
            metadata = dict()
        super().__init__(metadata=metadata)
        self.s3 = boto3.client(**kwargs)

        # Unique key specifying processing of a run
        self.strax_unique_key = key

        self.config = boto3.s3.transfer.TransferConfig(
            max_concurrency=40,
            num_download_attempts=30)

    def _save_chunk(self, data, chunk_info, executor=None):
        # Keyname
        key_name = f"{self.strax_unique_key}/{chunk_info['chunk_i']:06d}"

        # Save chunk via temporary file
        with tempfile.SpooledTemporaryFile() as f:
            filesize = strax.save_file(f,
                                       data=data,
                                       compressor=self.md['compressor'])
            f.seek(0)
            self.s3.upload_fileobj(f,
                                   BUCKET_NAME,
                                   key_name,
                                   Config=self.config)

        return dict(key_name=key_name, filesize=filesize), None

    def _save_chunk_metadata(self, chunk_info):
        self._upload_json(chunk_info,
                          f"{self.strax_unique_key}"
                          f"/metadata_{chunk_info['chunk_i']:06d}.json")

    def _close(self):
        # Collect all the chunk metadata
        prefix = f'{self.strax_unique_key}/metadata_'
        objects_list = self.s3.list_objects(Bucket=BUCKET_NAME,
                                            Prefix=prefix)

        if 'Contents' in objects_list:
            for file in objects_list['Contents']:
                # Grab chunk metadata as ASCIII
                result = self.s3.get_object(Bucket=BUCKET_NAME,
                                            Key=file['Key'])
                text = result["Body"].read().decode()

                # Save dictionary with central metadata
                self.md['chunks'].append(json.loads(text))

                # Delete chunk metadata
                self.s3.delete_object(Bucket=BUCKET_NAME,
                                      Key=file['Key'])

        # And make one run-wide metadata
        self._upload_json(self.md,
                          f'{self.strax_unique_key}/metadata.json')

    def _upload_json(self, document, filename):
        with tempfile.SpooledTemporaryFile() as f:
            text = json.dumps(document, **self.json_options)
            f.write(text.encode())
            f.seek(0)
            self.s3.upload_fileobj(f,
                                   BUCKET_NAME,
                                   filename)
