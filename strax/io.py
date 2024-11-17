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

# use tqdm as loaded in utils (from tqdm.notebook when in a jupyter env)
tqdm = strax.utils.tqdm

blosc.set_releasegil(True)
blosc.set_nthreads(1)

# zstd's default compression level is 3:
# https://github.com/sergey-dryabzhinsky/python-zstd/blob/eba9e633e0bc0e9c9762c985d0433e08405fd097/src/python-zstd.h#L53
# we also need to constraint the number of worker threads to 1
# https://github.com/sergey-dryabzhinsky/python-zstd/blob/eba9e633e0bc0e9c9762c985d0433e08405fd097/src/python-zstd.h#L98
zstd_compress = lambda data: zstd.compress(data, 3, 1)


COMPRESSORS = dict(
    bz2=dict(compress=bz2.compress, decompress=bz2.decompress),
    zstd=dict(compress=zstd_compress, decompress=zstd.decompress),
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
def save_file(f, data, compressor="zstd"):
    """Save data to file and return number of bytes written.

    :param f: file name or handle to save to
    :param data: data (numpy array) to save
    :param compressor: compressor to use

    """
    if isinstance(f, str):
        final_fn = f
        temp_fn = f + "_temp"
        with open(temp_fn, mode="wb") as write_file:
            result = _save_file(write_file, data, compressor)
        os.rename(temp_fn, final_fn)
        return result
    else:
        return _save_file(f, data, compressor)


def _save_file(f, data, compressor="zstd"):
    assert isinstance(data, np.ndarray), "Please pass a numpy array"
    d_comp = COMPRESSORS[compressor]["compress"](data)
    f.write(d_comp)
    return len(d_comp)


def _compress_blosc(data):
    if data.nbytes >= blosc.MAX_BUFFERSIZE:
        raise ValueError("Blosc's input buffer cannot exceed ~2 GB")
    return blosc.compress(data, shuffle=False)


COMPRESSORS["blosc"]["compress"] = _compress_blosc


@export
def dry_load_files(dirname, chunk_numbers=None, disable=False, **kwargs):
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
    results = np.hstack(results)
    return results if len(results) else np.empty(0, dtype)
