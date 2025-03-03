"""Read/write numpy arrays to/from compressed files or file-like objects."""

import os
import bz2
import json

import numpy as np
import blosc
import zstd
import zstandard
import lz4.frame as lz4
from ast import literal_eval
from io import BytesIO
from botocore.exceptions import ClientError

import strax

export, __all__ = strax.exporter()
__all__.extend(["DECOMPRESS_BUFFER_SIZE"])

DECOMPRESS_BUFFER_SIZE = 64 * 1024 * 1024  # 64 MB

# use tqdm as loaded in utils (from tqdm.notebook when in a jupyter env)
tqdm = strax.utils.tqdm

blosc.set_releasegil(True)
blosc.set_nthreads(1)


def _bz2_decompress(f, buffer_size=DECOMPRESS_BUFFER_SIZE):
    decompressor = bz2.BZ2Decompressor()
    data = bytearray()  # Efficient mutable storage
    for d in iter(lambda: f.read(buffer_size), b""):
        data.extend(decompressor.decompress(d))
    return data


# zstd's default compression level is 3:
# https://github.com/sergey-dryabzhinsky/python-zstd/blob/eba9e633e0bc0e9c9762c985d0433e08405fd097/src/python-zstd.h#L53
# we also need to constraint the number of worker threads to 1
# https://github.com/sergey-dryabzhinsky/python-zstd/blob/eba9e633e0bc0e9c9762c985d0433e08405fd097/src/python-zstd.h#L98
_zstd_compress = lambda data: zstd.compress(data, 3, 1)


def _zstd_decompress(f, chunk_size=64 * 1024 * 1024):
    decompressor = zstandard.ZstdDecompressor().decompressobj()
    data = bytearray()  # Efficient mutable storage
    for d in iter(lambda: f.read(chunk_size), b""):
        data.extend(decompressor.decompress(d))
    return data


def _blosc_compress(data):
    if data.nbytes >= blosc.MAX_BUFFERSIZE:
        raise ValueError("Blosc's input buffer cannot exceed ~2 GB")
    return blosc.compress(data, shuffle=False)


def _blosc_decompress(f):
    data = f.read()
    data = blosc.decompress(data)
    return data


def _lz4_decompress(f, buffer_size=DECOMPRESS_BUFFER_SIZE):
    decompressor = lz4.LZ4FrameDecompressor()
    data = bytearray()  # Efficient mutable storage
    for d in iter(lambda: f.read(buffer_size), b""):
        data.extend(decompressor.decompress(d))
    return data


COMPRESSORS = dict(
    bz2=dict(compress=bz2.compress, decompress=bz2.decompress, _decompress=_bz2_decompress),
    zstd=dict(compress=_zstd_compress, decompress=zstd.decompress, _decompress=_zstd_decompress),
    blosc=dict(
        compress=_blosc_compress, decompress=blosc.decompress, _decompress=_blosc_decompress
    ),
    lz4=dict(compress=lz4.compress, decompress=lz4.decompress, _decompress=_lz4_decompress),
)


@export
def load_file(f, compressor, dtype, S3_client=None, bucket_name=None, is_s3_path=False):
    """Read and return data from file or S3.

    :param f: file name or handle to read from
    :param compressor: compressor to use for decompressing. If not passed, will try to load it from
        json metadata file.
    :param dtype: numpy dtype of data to load
    :param S3_client: (optional) S3 client to find and store data
    :param is_s3_path: Boolean indicating if the file is stored in S3.

    """
    if is_s3_path:
        # Read from S3
        return load_file_from_s3(f, compressor, dtype, bucket_name)
    elif isinstance(f, str):
        # Read from local file
        with open(f, mode="rb") as read_file:
            return _load_file(read_file, compressor, dtype)
    else:
        # If f is already a file-like object, just use it
        return _load_file(f, compressor, dtype)


def load_file_from_s3(key, compressor, dtype, bucket_name):
    """Helper function to load data from S3."""
    s3 = strax.S3Frontend().s3

    try:
        data = COMPRESSORS[compressor]["_decompress"](key)
        if not len(data):
            return np.zeros(0, dtype=dtype)

        # Retrieve the file from S3 and load into a BytesIO buffer
        response = s3.get_object(Bucket=bucket_name, Key=key)
        file_data = response["Body"].read()  # Read the content of the file from S3

        # Create a file-like object from the binary data
        file_buffer = BytesIO(file_data)
        return _load_file(file_buffer, compressor, dtype)

    except ClientError as e:
        raise RuntimeError(f"Failed to load {key} from bucket {bucket_name}: {e}")


def _load_file(f, compressor, dtype):
    try:
        data = COMPRESSORS[compressor]["_decompress"](f)
        if not len(data):
            return np.zeros(0, dtype=dtype)
        try:
            return np.frombuffer(data, dtype=dtype)
        except ValueError as e:
            raise ValueError(f"ValueError while loading data with dtype =\n\t{dtype}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")

    except Exception:
        raise strax.DataCorrupted(
            f"Fatal Error while reading file {f}: " + strax.utils.formatted_exception()
        )


@export
def save_file(f, data, compressor="zstd", is_s3_path=False):
    """Save data to file and return number of bytes written.

    :param f: file name or handle to save to
    :param data: data (numpy array) to save
    :param compressor: compressor to use

    """

    if isinstance(f, str):
        final_fn = f
        temp_fn = f + "_temp"
        if is_s3_path is False:
            with open(temp_fn, mode="wb") as write_file:
                result = _save_file(write_file, data, compressor)
            os.rename(temp_fn, final_fn)
            return result
        else:
            s3_interface = strax.S3Frontend(
                s3_access_key_id=None,
                s3_secret_access_key=None,
                path="",
                deep_scan=False,
            )
            # Copy temp file to final file
            result = _save_file_to_s3(s3_interface, temp_fn, data, compressor)
            s3_interface.s3.copy_object(
                Bucket=s3_interface.BUCKET,
                Key=final_fn,
                CopySource={"Bucket": s3_interface.BUCKET, "Key": temp_fn},
            )

            # Delete the temporary file
            s3_interface.s3.delete_object(Bucket=s3_interface.BUCKET, Key=temp_fn)

            return result
    else:
        return _save_file(f, data, compressor)


def _save_file(f, data, compressor="zstd"):
    assert isinstance(data, np.ndarray), "Please pass a numpy array"
    d_comp = COMPRESSORS[compressor]["compress"](data)
    f.write(d_comp)
    return len(d_comp)


def _save_file_to_s3(s3_client, key, data, compressor=None):
    # Use this method to save file directly to S3
    # If compression is needed, handle it here
    # Use `BytesIO` to handle binary data in-memory
    assert isinstance(data, np.ndarray), "Please pass a numpy array"

    # Create a binary buffer to simulate writing to a file
    buffer = BytesIO()

    # Simulate saving file content (you can compress or directly write data here)
    if compressor:
        data = COMPRESSORS[compressor]["compress"](data)
    buffer.write(data)
    buffer.seek(0)  # Reset the buffer to the beginning

    # Upload buffer to S3 under the specified key
    s3_client.s3.put_object(Bucket=s3_client.BUCKET, Key=key, Body=buffer.getvalue())

    return len(data)


def _compress_blosc(data):
    if data.nbytes >= blosc.MAX_BUFFERSIZE:
        raise ValueError("Blosc's input buffer cannot exceed ~2 GB")
    return blosc.compress(data, shuffle=False)


COMPRESSORS["blosc"]["compress"] = _compress_blosc


@export
def dry_load_files(dirname, chunk_numbers=None, disable=False, **kwargs):
    prefix = strax.storage.files.dirname_to_prefix(dirname)
    metadata_json = f"{prefix}-metadata.json"
    md_path = os.path.join(dirname, metadata_json)

    with open(md_path, mode="r") as f:
        metadata = json.loads(f.read())

    dtype = literal_eval(metadata["dtype"])

    def load_chunk(chunk_info):
        if chunk_info["n"] != 0:
            data = load_file(
                os.path.join(dirname, f"{prefix}-{chunk_info['chunk_i']:06d}"),
                metadata["compressor"],
                dtype,
            )
            if len(data) != chunk_info["n"]:
                raise ValueError(
                    f"Chunk {chunk_info['chunk_i']:06d} has {len(data)} "
                    f"items, but metadata says {chunk_info['n']}."
                )
        return data if len(data) else np.empty(0, dtype)

    # Load all chunks if chunk_numbers is None, otherwise load the specified chunk
    if chunk_numbers is None:
        chunk_numbers = list(range(len(metadata["chunks"])))
    else:
        if not isinstance(chunk_numbers, (int, list, tuple)):
            raise ValueError(
                f"Chunk number must be int, list, or tuple, not {type(chunk_numbers)}."
            )
        chunk_numbers = (
            chunk_numbers if isinstance(chunk_numbers, (list, tuple)) else [chunk_numbers]
        )
        if max(chunk_numbers) >= len(metadata["chunks"]):
            raise ValueError(f"Chunk {max(chunk_numbers):06d} does not exist in {dirname}.")

    results = []
    for c in tqdm(chunk_numbers, disable=disable):
        chunk_info = metadata["chunks"][c]
        x = load_chunk(chunk_info)
        x = strax.apply_selection(x, **kwargs)
        results.append(x)

    # No need to hstack if only one chunk is loaded
    if len(results) == 1:
        results = results[0]
    else:
        results = np.hstack(results)
    return results if len(results) else np.empty(0, dtype)
