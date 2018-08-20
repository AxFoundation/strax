import glob
import os
import os.path as osp
import json
import shutil
import boto3
import tempfile

import strax
from .common import StorageFrontend

export, __all__ = strax.exporter()

@export
class SimpleS3Store(StorageFrontend):
    """Simplest registry: single directory with FileStore data
    sitting in subdirectories.

    Run-level metadata is stored in loose json files in the directory.
    """
    def __init__(self,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 endpoint_url='http://ceph-s3.mwt2.org',
                 bucket_name='meta',
                 run_metadata_filename = 'run_%s_metadata.json',
                 *args, **kwargs):
        """
        :param path: Path to folder with data subfolders.
        For other arguments, see DataRegistry base class.
        """
        super().__init__(*args, **kwargs)

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
        s3client.create_bucket(Bucket=bucket_name)  # Does nothing if exists

        self.backends = [S3Backend(aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key,
                                   endpoint_url=endpoint_url)]

        # Save for later
        self.bucket_name = bucket_name
        self.run_metadata_filename = run_metadata_filename


    def run_metadata(self, run_id):
        result = self.s3client.get_object(Bucket=self.bucket_name,
                                          Key=self.run_metadata_filename % run_id)
        text = result["Body"].read().decode()
        return json.loads(text)

    def write_run_metadata(self, run_id, metadata):
        self.s3client.put_object(Bucket=self.bucket_name,
                                 Key=self.run_metadata_filename % run_id,
                                 Body=json.dumps(metadata).encode())

    def _find(self, key, write, ignore_versions, ignore_config):

        # Check exact match / write case
        dirname = str(key)
        bk = self.backend_key(dirname)
        for key in self.s3client.list_objects(Bucket=self.bucket_name)['Contents']:
            if key['Key'] == dirname:
                if write:
                    if self._can_overwrite(key):
                        return bk
                    raise strax.DataExistsError(at=bk)
                return bk
        if write:
            return bk

        raise strax.DataNotAvailable

    def backend_key(self, dirname):
        return self.backends[0].__class__.__name__, dirname

    def remove(self, key):
        # There is no database, so removing the folder from the filesystem
        # (which FileStore should do) is sufficient.
        pass


@export
class S3Backend(strax.StorageBackend):
    """Store data locally in a directory of binary files.

    Files are named after the chunk number (without extension).
    Metadata is stored in a file called metadata.json.
    """
    def __init__(self, **kwargs):
        #  Initialized connection to S3-protocol storage
        self.s3client = boto3.client(**kwargs,
                                     service_name='s3')
        self.kwargs = kwargs

    def get_metadata(self, dirname):
        result = self.s3client.get_object(Bucket=dirname,
                                          Key='metadata.json')

        text = result["Body"].read().decode()
        return json.loads(text)

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        with tempfile.SpooledTemporaryFile() as f:
            self.s3client.download_fileobj(Bucket=dirname,
                                           Key=chunk_info['filename'],
                                           Fileobj=f)
            return strax.load_file(f, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, meta_only=False):
        return S3Saver(dirname, metadata=metadata, meta_only=meta_only,
                       **self.kwargs)


@export
class S3Saver(strax.Saver):
    """Saves data to compressed binary files"""
    json_options = dict(sort_keys=True, indent=4)

    def __init__(self, dirname, metadata, meta_only,
                 **kwargs):
        super().__init__(metadata, meta_only)
        self.s3client = boto3.client(**kwargs,
                                     service_name='s3')

        self.dirname = dirname

        self.s3client.create_bucket(Bucket=self.dirname)  # Does nothing if exists



    def _save_chunk(self, data, chunk_info):
        filename = '%06d' % chunk_info['chunk_i']

        with tempfile.SpooledTemporaryFile() as f:
            filesize = strax.save_file(f,
                                       data=data,
                                       compressor=self.md['compressor'])
            self.s3client.upload_fileobj(f, self.dirname, filename)

        return dict(filename=filename, filesize=filesize)

    def _save_chunk_metadata(self, chunk_info):
        if self.meta_only:
            # TODO HACK!
            chunk_info["filename"] = '%06d' % chunk_info['chunk_i']

        with tempfile.SpooledTemporaryFile(mode='w') as f:
            f.write(json.dumps(chunk_info, **self.json_options))
            self.s3client.upload_fileobj(f, self.dirname, f'metadata_{chunk_info["filename"]}.json')

    #def _close(self):
    #    for fn in sorted(glob.glob(
    #            self.tempdirname + '/metadata_*.json')):
    #        with open(fn, mode='r') as f:
    #            self.md['chunks'].append(json.load(f))
    #        os.remove(fn)

    #    with open(self.tempdirname + '/metadata.json', mode='w') as f:
    #        f.write(json.dumps(self.md, **self.json_options))
    #    os.rename(self.tempdirname, self.dirname)
