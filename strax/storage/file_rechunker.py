import tempfile
import os
import typing
import time
import shutil
import strax
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

export, __all__ = strax.exporter()


@export
def rechunker(
    source_directory: str,
    dest_directory: typing.Optional[str] = None,
    replace: bool = False,
    compressor: typing.Optional[str] = None,
    target_size_mb: typing.Optional[str] = None,
    rechunk: bool = True,
    progress_bar: bool = True,
    parallel: typing.Union[bool, str] = False,
    max_workers: int = 4,
    _timeout: int = 24 * 3600,
) -> dict:
    """Rechunk/Recompress a strax-datatype saved in a FileSystemBackend outside the strax.Context.
    For a user-friendly context centered alternative, see strax.Context.copy_to_frontend: github.com
    /AxFoundation/strax/blob/a0d51fdd3bea52c228c8f74c614fc77bb7cf1bc5/strax/context.py#L1792  #
    noqa.

    One can either specify a destination directory where to store a new
    copy of this data with <dest_directory> or replace the input file
    with it's rechunked version.

    This function can either:
     - rechunk (if <rechunk? is True), probably incrementing the
        <target_size_mb> is also useful (create larger chunks)
     - recompress (if a <compressor> is specified)

    One can also rechunk and recompress simultaneously

    :param source_directory: Path to a folder containing a single
        strax.DataType.
    :param dest_directory: Head of a folder whereto write new data. If
        nothing is specified, write to a temporary directory.
    :param replace: Delete the source_directory and replace it with it's
        rechunked version
    :param compressor: Compressor to be used in saving the rechunked
        data
    :param target_size_mb: Target size of chunks (uncompressed). As long
        as a chunk is smaller than this many MB, keep adding new MBs
        until the chunk is at least target_size_mb or we run out of
        chunks.
    :param rechunk: Do we want to rechunk?
    :param progress_bar: display progress bar
    :param parallel: Parallelize using threadpool or process pool (otherwise run in serial)
    :param max_workers: number of threads/cores to associate to the parallel
        processing, only relevant when parallel is specified
    :param _timeout: Mailbox timeout
    :return: Dictionary with some information on the write/load times
        involved.

    """
    _check_arguments(source_directory, replace, dest_directory, parallel)
    backend_key = os.path.basename(os.path.normpath(source_directory))
    dest_directory, _temp_dir = _get_dest_and_tempdir(dest_directory, replace, backend_key)

    backend = strax.FileSytemBackend(set_target_chunk_mb=target_size_mb)
    meta_data, source_compressor = _get_meta_data_and_compressor(
        backend, source_directory, compressor, target_size_mb
    )
    executor = _get_executor(parallel, max_workers)

    data_loader = backend.loader(source_directory, executor=executor)
    pbar = strax.utils.tqdm(
        total=len(meta_data["chunks"]), disable=not progress_bar, desc=backend_key
    )
    load_time_seconds = []

    def load_wrapper(generator):
        """Wrapped loader for bookkeeping load time."""
        n_bytes = 0
        while True:
            try:
                t0 = time.time()
                data = next(generator)
                t1 = time.time()
                load_time_seconds.append(t1 - t0)
                n_bytes += data.nbytes
                pbar.postfix = f"{(n_bytes / 1e6) / (t1 - pbar.start_t):.1f} MB/s"
                pbar.n += 1
                pbar.display()
            except StopIteration:
                pbar.close()
                return
            yield data

    print(f"Rechunking {source_directory} to {dest_directory}")
    saver = backend._saver(dest_directory, metadata=meta_data, saver_timeout=_timeout)

    write_time_start = time.time()
    _exhaust_generator(executor, saver, load_wrapper, data_loader, rechunk, _timeout)
    if not os.path.exists(dest_directory):  # type: ignore
        raise FileNotFoundError(f"{dest_directory} not found, did one of the savers die?")
    load_time = sum(load_time_seconds)
    write_time = time.time() - write_time_start - load_time

    _move_directories(replace, source_directory, dest_directory, _temp_dir)

    return dict(
        backend_key=backend_key,
        load_time=load_time,
        write_time=write_time,
        uncompressed_mb=sum([x["nbytes"] for x in meta_data["chunks"]]) / 1e6,
        source_compressor=source_compressor,
        dest_compressor=meta_data["compressor"],
        dest_directory=dest_directory,
    )


def _check_arguments(source_directory, replace, dest_directory, parallel):
    if not os.path.exists(source_directory):
        raise FileNotFoundError(f"No file at {source_directory}")
    if not replace and dest_directory is None:
        raise ValueError(
            f"Specify a destination path <dest_file> when not replacing the original path"
        )
    if parallel not in [False, True, "thread", "process"]:
        raise ValueError('Choose from False, "thread" or "process"')


def _get_dest_and_tempdir(dest_directory, replace, backend_key):
    if dest_directory is None and replace:
        _temp_dir = tempfile.TemporaryDirectory()
        dest_directory = _temp_dir.name
    else:
        _temp_dir = False

    if os.path.basename(os.path.normpath(dest_directory)) != backend_key:
        # New key should be correct! If there is not an exact match,
        # we want to make sure that we append the backend_key correctly
        print(f"Will write to {dest_directory} and make sub-folder {backend_key}")
        dest_directory = os.path.join(dest_directory, backend_key)
    return dest_directory, _temp_dir


def _move_directories(replace, source_directory, dest_directory, _temp_dir):
    if replace:
        print(f"move {dest_directory} to {source_directory}")
        shutil.rmtree(source_directory)
        shutil.move(dest_directory, source_directory)
    if _temp_dir:
        _temp_dir.cleanup()


def _exhaust_generator(executor, saver, load_wrapper, data_loader, rechunk, _timeout):
    if executor is None:
        saver.save_from(load_wrapper(data_loader), rechunk=rechunk)
        return

    mailbox = strax.Mailbox(name="rechunker", timeout=_timeout)
    mailbox.add_sender(data_loader)
    mailbox.add_reader(
        partial(
            saver.save_from,
            executor=executor,
            rechunk=rechunk,
        )
    )
    final_generator = mailbox.subscribe()

    # Make sure everything is added to the mailbox before starting!
    mailbox.start()
    for _ in load_wrapper(final_generator):
        pass

    mailbox.cleanup()
    executor.shutdown(wait=True)


def _get_meta_data_and_compressor(backend, source_directory, compressor, target_size_mb):
    meta_data = backend.get_metadata(source_directory)
    old_compressor = meta_data["compressor"]
    if compressor is not None:
        meta_data["compressor"] = compressor
    if target_size_mb is not None:
        meta_data["chunk_target_size_mb"] = target_size_mb
    return meta_data, old_compressor


def _get_executor(parallel, max_workers):
    return {
        True: ThreadPoolExecutor(max_workers),
        "thread": ThreadPoolExecutor(max_workers),
        "process": (
            ProcessPoolExecutor(max_workers)
            if strax.SHMExecutor is None
            else strax.SHMExecutor(max_workers)
        ),
    }.get(parallel)
