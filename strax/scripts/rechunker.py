import os.path
import argparse

import strax
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rechunker for FileSytemBackend. Interfaces with strax.rechunker."
        "Please see the documentation of strax.rechunker for more information: "
        "https://github.com/AxFoundation/strax/blob/31c114c5f8329e53289d5127fb2125e71c3d6aae/strax/storage/files.py#L371"  # noqa
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Target directory to rechunk, should be a folder in a "
        "strax.DataDirectory (one datatype)",
    )
    parser.add_argument(
        "--dest",
        "--destination",
        default=None,
        dest="dest",
        type=str,
        help="Where to store rechunked data. If nothing is specified, replace the source.",
    )
    parser.add_argument(
        "--compressor",
        choices=list(strax.io.COMPRESSORS.keys()),
        help="Recompress using one of these compressors. If nothing specified, "
        "use the same compressor as the source",
    )
    parser.add_argument(
        "--rechunk", default=True, choices=[True, False], type=bool, help="rechunk the data"
    )
    parser.add_argument(
        "--target_size_mb",
        "--target-size-mb",
        dest="target_size_mb",
        type=int,
        default=strax.DEFAULT_CHUNK_SIZE_MB,
        help="Target size MB (uncompressed) of the rechunked data",
    )
    parser.add_argument(
        "--write_stats_to",
        "--write-stats-to",
        dest="write_stats_to",
        type=str,
        default=None,
        help="Write some information to this file (csv format)",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default="False",
        choices=["False", "True", "thread", "process"],
        help="Parallelize using threadpool or processpool",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers if parallel is specified"
    )
    parser.add_argument("--profile_memory", action="store_true", help="Profile memory usage")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.profile_memory:
        from memory_profiler import memory_usage
        import time

        start = time.time()
        mem = memory_usage(proc=(rechunk, (args,)))
        print(f"Memory profiler says peak RAM usage was: {max(mem):.1f} MB")
        print(f"Took {time.time() - start:.1f} s = {(time.time() - start) / 3600:.2f} h ")
        print("Bye, bye")
    else:
        rechunk(args)


def rechunk(args):
    source_mb = strax.utils.dir_size_mb(args.source)
    report = strax.rechunker(
        source_directory=args.source,
        dest_directory=args.dest,
        replace=args.dest is None,
        compressor=args.compressor,
        target_size_mb=args.target_size_mb,
        rechunk=args.rechunk,
        parallel={"False": False, "True": True}.get(args.parallel, args.parallel),
        max_workers=args.max_workers,
    )
    if args.dest is not None:
        recompressed_mb = strax.utils.dir_size_mb(report.get("dest_directory", args.dest))
    else:
        recompressed_mb = strax.utils.dir_size_mb(args.source)
    report.update(dict(source_mb=source_mb, dest_mb=recompressed_mb))
    if args.write_stats_to:
        if os.path.exists(args.write_stats_to):
            df = pd.read_csv(args.write_stats_to)
        else:
            df = pd.DataFrame()
        df_new = pd.concat([df, pd.DataFrame({k: [v] for k, v in report.items()})])
        df_new.to_csv(args.write_stats_to, index=False)

    print(f"Re-compressed {args.source}")
    for k, v in report.items():
        print(f"\t{k:16}\t{v}")


if __name__ == "__main__":
    main()
