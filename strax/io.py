"""Read/write numpy arrays to/from compressed files or file-like objects."""

import os
import bz2
import json

import numpy as np
import blosc
import zstd
import lz4.frame as lz4
from ast import literal_eval

import strax
from strax import RUN_METADATA_PATTERN

export, __all__ = strax.exporter()

blosc.set_releasegil(True)


COMPRESSORS = dict(
    bz2=dict(compress=bz2.compress, decompress=bz2.decompress),
    zstd=dict(compress=zstd.compress, decompress=zstd.decompress),
    blosc=dict(
        compress=None,  # add special function to prevent overflow at bottom module
        decompress=blosc.decompress,
    ),
    lz4=dict(compress=lz4.compress, decompress=lz4.decompress),
)


@export
def load_file(f, compressor, dtype):
    """Read and return data from file.

    :param f: file name or handle to read from
    :param compressor: compressor to use for decompressing. If not passed, will try to load it from
        json metadata file.
    :param dtype: numpy dtype of data to load

    """
    if isinstance(f, str):
        with open(f, mode="rb") as write_file:
            return _load_file(write_file, compressor, dtype)
    else:
        return _load_file(f, compressor, dtype)


def _load_file(f, compressor, dtype):
    try:
        data = f.read()
        if not len(data):
            return np.zeros(0, dtype=dtype)

        data = COMPRESSORS[compressor]["decompress"](data)
        try:
            return np.frombuffer(data, dtype=dtype)
        except ValueError as e:
            raise ValueError(f"ValueError while loading data with dtype =\n\t{dtype}") from e

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
def dry_load_files(dirname, chunk_number=None):
    prefix = strax.storage.files.dirname_to_prefix(dirname)
    metadata_json = RUN_METADATA_PATTERN % prefix
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

    # Load all chunks if chunk_number is None, otherwise load the specified chunk
    if chunk_number is None:
        chunk_numbers = list(range(len(metadata["chunks"])))
    else:
        if not isinstance(chunk_number, int):
            raise ValueError(f"Chunk number must be an integer, not {chunk_number}.")
        if chunk_number >= len(metadata["chunks"]):
            raise ValueError(f"Chunk {chunk_number:06d} does not exist in {dirname}.")
        chunk_numbers = [chunk_number]

    results = []
    for c in chunk_numbers:
        chunk_info = metadata["chunks"][c]
        results.append(load_chunk(chunk_info))
    results = np.hstack(results)
    return results if len(results) else np.empty(0, dtype)
