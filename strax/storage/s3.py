"""I/O that speaks the S3 protocol

The S3 protocol is an HTTP-based protocol spoken by Amazon Web Services, but
also storage systems such as CEPH (used in particle physics).  Therefore,
this can be widely used if you know the appropriate endpoint.

Be aware that you must specify the following two environmental variables or
pass the appropriate valeus to the constructor:

  *  AWS_ACCESS_KEY_ID
  *  AWS_SECRET_ACCESS_KEY

"""

import json
import os
import tempfile

import boto3

import strax
from strax import StorageFrontend

export, __all__ = strax.exporter()


@export
class SimpleS3Store(StorageFrontend):
    """Frontend for S3 stores that just checks if data exists.

    This uses boto3 for communicating, where you can look at their docs
    to understand a lot of this.  S3 is an object store where each object is
    chunk.  The bucket corresponds to strax key (run / plugin).

    Currently, no run level metadata is stored.
    """

    def __init__(self,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 endpoint_url='http://ceph-s3.mwt2.org',
                 *args, **kwargs):
        """
        You must provide credentials to access your storage element.

        :param aws_access_key_id: access key for S3-readable storage.
        :param aws_secret_access_key: secret key for S3-readable storage.
        :param endpoint_url: URL of S3-readable storage.

        For other arguments, see DataRegistry base class.
        """

        super().__init__(*args, **kwargs)

        # Get S3 protocol credentials
        if aws_access_key_id is None:
            if 'AWS_ACCESS_KEY_ID' not in os.environ:
                raise EnvironmentError("S3 access key not specified")
            aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        if aws_secret_access_key is None:
            if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
                raise EnvironmentError("S3 secret key not specified")
            aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

        #  Initialized connection to S3-protocol storage
        self.s3client = boto3.client(aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key,
                                     endpoint_url=endpoint_url,
                                     service_name='s3')

        # Setup backends for reading
        self.backends = [S3Backend(aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key,
                                   endpoint_url=endpoint_url)]

    def _find(self, key, write, ignore_versions, ignore_config):
        """Determine if data exists

        Search the S3 store to see if data is there.
        """
        # Check exact match / write case
        key_str = str(key)
        bk = self.backend_key(key_str)

        objects_list = self.s3client.list_buckets()
        if 'Buckets' not in objects_list:
            # No objects yet... so no metadata
            if write:
                # If write, it can be that no metadata exists
                return bk
            else:
                # If reading and no objects, then problem
                raise strax.DataNotAvailable

        # Loop over all buckets to look for our data
        for key in objects_list['Buckets']:
            if key['Name'] == key_str:
                if write:
                    if self._can_overwrite(key):
                        return bk
                    raise strax.DataExistsError(at=bk)
                return bk
        raise strax.DataNotAvailable

    def backend_key(self, key_str):
        return self.backends[0].__class__.__name__, key_str

    def remove(self, key):
        pass


@export
class S3Backend(strax.StorageBackend):
    """Store data in S3-backend

    Buckets are run/plugin keys and objects are chunks.
    """

    def __init__(self, **kwargs):
        super().__init__()

        #  Initialized connection to S3-protocol storage
        self.s3client = boto3.client(**kwargs,
                                     service_name='s3')
        self.kwargs = kwargs

    def get_metadata(self, backend_key):
        result = self.s3client.get_object(Bucket=backend_key,
                                          Key='metadata.json')

        text = result["Body"].read().decode()
        return json.loads(text)

    def _read_chunk(self, backend_key, chunk_info, dtype, compressor):
        with tempfile.SpooledTemporaryFile() as f:
            self.s3client.download_fileobj(Bucket=backend_key,
                                           Key=chunk_info['filename'],
                                           Fileobj=f)
            f.seek(0)  # Needed?
            return strax.load_file(f, dtype=dtype, compressor=compressor)

    def _saver(self, key, metadata, meta_only=False):
        return S3Saver(key, metadata=metadata, meta_only=meta_only,
                       **self.kwargs)


@export
class S3Saver(strax.Saver):
    """Saves data to S3-compatible storage
    """
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, key, metadata, meta_only,
                 **kwargs):
        super().__init__(metadata, meta_only)
        self.s3client = boto3.client(**kwargs,
                                     service_name='s3')

        # Unique key specifying processing of a run
        self.key = key

        self.s3client.create_bucket(Bucket=self.key)  # Does nothing if exists

    def _save_chunk(self, data, chunk_info):
        filename = '%06d' % chunk_info['chunk_i']

        with tempfile.SpooledTemporaryFile() as f:
            filesize = strax.save_file(f,
                                       data=data,
                                       compressor=self.md['compressor'])
            f.seek(0)
            self.s3client.upload_fileobj(f, self.key, filename)

        return dict(filename=filename, filesize=filesize)

    def _save_chunk_metadata(self, chunk_info):
        if self.meta_only:
            # TODO HACK!
            chunk_info["filename"] = '%06d' % chunk_info['chunk_i']

        with tempfile.SpooledTemporaryFile() as f:
            f.write(json.dumps(chunk_info, **self.json_options).encode())
            f.seek(0)
            self.s3client.upload_fileobj(f, self.key, f'metadata_{chunk_info["filename"]}.json')

    def _close(self):
        # Collect all the chunk metadata
        objects_list = self.s3client.list_objects(Bucket=self.key)
        if 'Contents' in objects_list:
            for file in objects_list['Contents']:
                if not file['Key'].startswith('metadata_'):
                    continue

                result = self.s3client.get_object(Bucket=self.key,
                                                  Key=file['Key'])

                text = result["Body"].read().decode()

                self.md['chunks'].append(json.loads(text))

                self.s3client.delete_object(Bucket=self.key,
                                            Key=file['Key'])

        # And make one run-wide metadata
        with tempfile.SpooledTemporaryFile() as f:
            f.write(json.dumps(self.md, **self.json_options).encode())
            f.seek(0)
            self.s3client.upload_fileobj(f, self.key, f'metadata.json')
