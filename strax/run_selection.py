"""Context methods dealing with run scanning and selection."""

import fnmatch
import re
import typing as ty
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import pytz
import datetime
import strax
from strax import stable_argsort

# use tqdm as loaded in utils (from tqdm.notebook when in a jupyter env)
tqdm = strax.utils.tqdm

export, __all__ = strax.exporter()


@strax.Context.add_method
def list_available(self, target, runs=None, **kwargs) -> list:
    """Return sorted list of run_id's for which target is available.

    :param target: Data type to check
    :param runs: Runs to check. If None, check all runs.

    """

    if len(kwargs):
        # noinspection PyMethodFirstArgAssignment
        self = self.new_context(**kwargs)

    if self.runs is None:
        self.scan_runs()

    if runs is None:
        runs = self.runs["name"].values  # type: ignore
    else:
        runs = strax.to_str_tuple(runs)

    keys = set(self.keys_for_runs(target, runs))

    found: ty.Set[strax.DataKey] = set()
    for sf in self.storage:
        remaining = keys - found
        is_found = sf.find_several(list(remaining), **self._find_options)
        found |= set([k for i, k in enumerate(remaining) if is_found[i]])
    return list(sorted([x.run_id for x in found]))


@strax.Context.add_method
def keys_for_runs(
    self, target: str, run_ids: ty.Union[np.ndarray, list, tuple, str]
) -> ty.List[strax.DataKey]:
    """Get the data-keys for a multitude of runs. If use_per_run_defaults is False which it
    preferably is (#246), getting many keys should be fast as we only compute the lineage once.

    :param run_ids: Runs to get datakeys for
    :param target: datatype requested
    :return: list of datakeys of the target for the given runs.

    """
    run_ids = strax.to_str_tuple(run_ids)

    if self.context_config["use_per_run_defaults"]:
        return [self.key_for(r, target) for r in run_ids]
    elif len(run_ids):
        # Get the lineage once, for the context specifies that the
        # defaults may not change!
        p = self._get_plugins((target,), run_ids[0])[target]
        return [self.get_data_key(r, target, p.lineage) for r in run_ids]
    else:
        return []


@strax.Context.add_method
def scan_runs(
    self: strax.Context,
    check_available=tuple(),
    if_check_available="raise",
    store_fields=tuple(),
) -> pd.DataFrame:
    """Update and return self.runs with runs currently available in all storage frontends.

    :param check_available: Check whether these data types are available Availability of xxx is
        stored as a boolean in the xxx_available column.
    :param if_check_available: 'raise' (default) or 'skip', whether to do the check
    :param store_fields: Additional fields from run doc to include as rows in the dataframe. The
        context options scan_availability and store_run_fields list data types and run fields,
        respectively, that will always be scanned.

    """
    store_fields = tuple(
        set(
            list(strax.to_str_tuple(store_fields))
            + ["name", "number", "tags", "mode", strax.RUN_DEFAULTS_KEY]
            + list(self.context_config["store_run_fields"])
        )
    )
    if if_check_available == "raise":
        check_available = tuple(
            set(
                list(strax.to_str_tuple(check_available))
                + list(self.context_config["check_available"])
            )
        )
    elif if_check_available == "skip":
        check_available = tuple()
    else:
        raise ValueError(f"Invalid value for if_check_available: {if_check_available}")

    for target in check_available:
        save_when = self.get_save_when(target)
        if save_when < strax.SaveWhen.ALWAYS:
            p = self._plugin_class_registry[target]
            self.log.warning(
                f"{p.__name__}-plugin is {str(save_when)}. "
                f"Therefore {target} is most likely not stored!"
            )

    docs = None
    for sf in self.storage:
        _temp_docs = []
        for doc in sf._scan_runs(store_fields=store_fields):
            # If there is no number, make one from the name
            _is_superrun = doc.get("name", "not_a_superrun_if_no_name").startswith("_")
            if "number" not in doc:
                if "name" not in doc:
                    raise ValueError(f"Invalid run doc {doc}, contains neither name nor number.")
                if not _is_superrun:
                    # If there is no name, make one from the number
                    doc["number"] = int(doc["name"])
            if not _is_superrun:
                doc.setdefault("name", f"{doc['number']:06d}")

            # Convert tags/mode/source list to a,separated string if needed (mode and source can
            # be lists for superruns)
            doc.setdefault("mode", "")
            if isinstance(doc["mode"], list):
                doc["mode"] = ",".join(doc["mode"])

            doc.setdefault("source", "")
            if isinstance(doc["source"], list):
                doc["source"] = ",".join(doc["source"])

            doc["tags"] = ",".join(
                [t["name"] if isinstance(t, dict) else t for t in doc.get("tags", [])]
            )

            # Set a default livetime if we have start and stop
            if (
                "start" in store_fields
                and "end" in store_fields
                and "livetime" in store_fields
                and doc.get("start") is not None
                and doc.get("end") is not None
            ):
                doc.setdefault("livetime", doc["end"] - doc["start"])

            if _is_superrun:
                # In contrast to regular run-docs,
                # superruns are timezone aware. So strip off timezone
                # again:
                start = doc.get("start")
                if start:
                    doc["start"] = start.replace(tzinfo=None)

                start = doc.get("end")
                if start:
                    doc["end"] = start.replace(tzinfo=None)

                if type(doc.get("livetime")) == float:
                    # If we have a superrun livetime is stored as intger seconds
                    # as timedelta is not json serializable
                    doc["livetime"] = datetime.timedelta(seconds=doc["livetime"])

            # Put the strax defaults stuff into a different cache
            if strax.RUN_DEFAULTS_KEY in doc:
                self._run_defaults_cache[doc["name"]] = doc[strax.RUN_DEFAULTS_KEY]
                del doc[strax.RUN_DEFAULTS_KEY]

            doc = flatten_run_metadata(doc)

            _temp_docs.append(doc)

        if len(_temp_docs):
            new_docs = pd.DataFrame(_temp_docs)
        else:
            new_docs = pd.DataFrame([], columns=store_fields)

        if docs is None:
            docs = new_docs
        else:
            # Keep only new runs (not found by earlier frontends)
            mask = ~np.in1d(new_docs["name"], docs["name"])
            if np.any(mask):
                docs = pd.concat([docs, new_docs[mask]], sort=False)
                docs.reset_index(drop=True, inplace=True)

    # Rearrange columns
    if not self.context_config["use_per_run_defaults"] and strax.RUN_DEFAULTS_KEY in docs.columns:
        del docs[strax.RUN_DEFAULTS_KEY]
    docs = docs[["name"] + [x for x in docs.columns.tolist() if x != "name"]]  # type: ignore
    self.runs = docs

    # Add available data types,
    # this is kept for the case users directly call list_available
    for d in tqdm(
        check_available,
        desc="Checking data availability",
        disable=not len(check_available),
    ):
        self.runs[d + "_available"] = np.in1d(self.runs.name.values, self.list_available(d))

    return self.runs


@strax.Context.add_method
def select_runs(
    self,
    run_mode=None,
    run_id=None,
    include_tags=None,
    exclude_tags=None,
    available=tuple(),
    pattern_type="fnmatch",
    ignore_underscore=True,
    force_reload=False,
):
    """Return pandas.DataFrame with basic info from runs that match selection criteria.

    :param run_mode: Pattern to match run modes (reader.ini.name)
    :param run_id: Pattern to match a run_id or run_ids
    :param available: str or tuple of strs of data types for which data
        must be available according to the runs DB.
    :param include_tags: String or list of strings of patterns
        for required tags
    :param exclude_tags: String / list of strings of patterns
        for forbidden tags.
        Exclusion criteria have higher priority than inclusion criteria.
    :param pattern_type: Type of pattern matching to use.
        Defaults to 'fnmatch', which means you can use
        unix shell-style wildcards (`?`, `*`).
        The alternative is 're', which means you can use
        full python regular expressions.
    :param ignore_underscore: Ignore the underscore at the start of tags
        (indicating some degree of officialness or automation).
    :param force_reload: Force reloading of runs from storage.
        Otherwise, runs are cached after the first time they are loaded in self.runs.

    Examples:
     - `run_selection(include_tags='blinded')`
        select all datasets with a blinded or _blinded tag.
     - `run_selection(include_tags='*blinded')`
        ... with blinded or _blinded, unblinded, blablinded, etc.
     - `run_selection(include_tags=['blinded', 'unblinded'])`
        ... with blinded OR unblinded, but not blablinded.
     - `run_selection(include_tags='blinded',
                      exclude_tags=['bad', 'messy'])`
        ... select blinded dsatasets that aren't bad or messy

    """
    if self.runs is None or force_reload:
        self.scan_runs(if_check_available="skip")
    dsets = self.runs.copy()

    if pattern_type not in ("re", "fnmatch"):
        raise ValueError("Pattern type must be 're' or 'fnmatch'")

    # Filter datasets by run mode and/or name first
    for field_name, requested_value in (("name", run_id), ("mode", run_mode)):
        if requested_value is None:
            continue

        requested_value = strax.to_str_tuple(requested_value)

        values = dsets[field_name].values
        mask = np.zeros(len(values), dtype=bool)

        if pattern_type == "fnmatch":
            for i, x in enumerate(values):
                mask[i] = np.any([fnmatch.fnmatch(x, rv) for rv in requested_value])
        elif pattern_type == "re":
            for i, x in enumerate(values):
                mask[i] = np.any([re.match(rv, x) for rv in requested_value])

        dsets = dsets[mask]

    dsets = _include_exclude_tags(
        dsets,
        include_tags,
        exclude_tags,
        pattern_type,
        ignore_underscore,
    )

    # Check data availability only for selected datasets
    check_available = tuple(
        set(list(strax.to_str_tuple(available)) + list(self.context_config["check_available"]))
    )

    for d in tqdm(
        check_available,
        desc="Checking data availability",
        disable=not len(check_available),
    ):
        dsets[d + "_available"] = np.in1d(
            dsets.name.values, self.list_available(target=d, runs=dsets.name.values)
        )

    # This will help users call select_runs multiple times
    # with same run mode and/or name, but different available
    have_available = strax.to_str_tuple(available)
    for d in have_available:
        if not d + "_available" in dsets.columns:
            # Get extra availability info from the run db
            d_available = np.in1d(
                dsets.name.values, self.list_available(target=d, runs=dsets.name.values)
            )
            # Save both in the context and for this selection using
            # available = ('data_type',)
            dsets[d + "_available"] = d_available

    # Only return dsets available
    for d in have_available:
        dsets = dsets[dsets[d + "_available"]]

    return dsets


@strax.Context.add_method
def define_run(
    self: strax.Context,
    name: str,
    data: ty.Union[np.ndarray, pd.DataFrame, dict, list, tuple],
    from_run: ty.Union[str, None] = None,
):
    """Function for defining new superruns from a list of run_ids.

    Note:
        The function also allows to create a superrun from data
        (numpy.arrays/pandas.DataFframrs). However, this is currently
        not supported from the data loading side.

    :param name: Name/run_id of the superrun. Suoerrun names must start
        with an underscore.
    :param data: Data from which the superrun should be created. Can be
        either one of the following: a tuple/list of run_ids or a
        numpy.array/pandas.DataFrame containing some data.
    :param from_run: List of run_ids which were used to create the
        numpy.array/pandas.DataFrame passed in data.

    """
    if isinstance(data, (pd.DataFrame, np.ndarray)):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame.from_records(data)

        # strax.endtime does not work with DataFrames due to numba
        if "endtime" in data.columns:
            end = data["endtime"]
        else:
            end = data["time"] + data["length"] * data["dt"]

        # Array of events / regions of interest
        start, end = data["time"], end
        if from_run is not None:
            return self.define_run(name, {from_run: np.transpose([start, end])})
        elif "run_id" not in data.columns:
            raise ValueError(
                "Must provide from_run or data with a run_id column to define a superrun"
            )
        else:
            df = pd.DataFrame(dict(start=start, end=end, run_id=data["run_id"]))
            return self.define_run(
                name,
                {
                    run_id: rs[["start", "end"]].values.transpose()
                    for run_id, rs in df.groupby("run_id")
                },
            )

    if isinstance(data, (list, tuple)):
        # list of run_ids
        data = strax.to_str_tuple(data)
        return self.define_run(name, {run_id: "all" for run_id in data})

    if not isinstance(data, dict):
        raise ValueError(f"Can't define run from {type(data)}")

    # Find start and end time of the new run = earliest start time of other runs
    run_md = dict(
        start=datetime.datetime.max.replace(tzinfo=pytz.utc),
        end=datetime.datetime.min.replace(tzinfo=pytz.utc),
        livetime=0,
    )
    keys = []
    starts = []
    tags = set()
    modes = set()
    sources = set()
    comments = set()
    for _subrunid in data:
        doc = self.run_metadata(
            _subrunid, projection=["start", "end", "mode", "tags", "source", "comments"]
        )
        doc.setdefault(
            "tags",
            [
                {"name": ""},
            ],
        )
        doc.setdefault("mode", "")
        doc.setdefault("source", "")
        doc.setdefault(
            "comments",
            [
                {"comment": ""},
            ],
        )

        tags |= set([tag["name"] for tag in doc["tags"]])
        comments |= set([comment["comment"] for comment in doc["comments"]])

        modes |= set(strax.to_str_tuple(doc["mode"]))
        sources |= set(strax.to_str_tuple(doc["source"]))
        if len(sources) > 1:
            warnings.warn(f'You are defining a superrun with more than one source: "{sources}"')

        run_doc_start = doc["start"].replace(tzinfo=pytz.utc)
        run_doc_end = doc["end"].replace(tzinfo=pytz.utc)

        run_md["start"] = min(run_md["start"], run_doc_start)
        run_md["end"] = max(run_md["end"], run_doc_end)

        time_delta = run_doc_end - run_doc_start
        run_md["livetime"] += time_delta.total_seconds()
        keys.append(_subrunid)
        starts.append(run_doc_start)

    run_md["mode"] = tuple(modes)
    run_md["source"] = tuple(sources)
    run_md["tags"] = [{"name": tag} for tag in tags]
    run_md["comments"] = [{"comment": comment} for comment in comments]

    # Make sure subruns are sorted in time
    sort_index = stable_argsort(starts)
    data = {keys[i]: data[keys[i]] for i in sort_index}

    # Superrun names must start with an underscore
    if not name.startswith("_"):
        name = "_" + name
    # Dict mapping run_id: array of time ranges or all
    for sf in self.storage:
        if not sf.readonly and sf.can_define_runs:
            sf.define_run(name, sub_run_spec=data, **run_md)
            break
    else:
        raise RuntimeError("No storage frontend registered that allows run definition")


@strax.Context.add_method
def available_for_run(
    self: strax.Context,
    run_id: str,
    include_targets: ty.Union[None, list, tuple, str] = None,
    exclude_targets: ty.Union[None, list, tuple, str] = None,
    pattern_type: str = "fnmatch",
) -> pd.DataFrame:
    """For a given single run, check all the targets if they are stored. Excludes the target if
    never stored anyway.

    :param run_id: requested run
    :param include_targets: targets to include e.g. raw_records, raw_records* or *_nv. If multiple
        targets (e.g. a list) is provided, the target should match any of the arguments!
    :param exclude_targets: targets to exclude e.g. raw_records, raw_records* or *_nv. If multiple
        targets (e.g. a list) is provided, the target should match none of the arguments!
    :param pattern_type: either 'fnmatch' (Unix filename pattern matching) or 're' (Regular
        expression operations).
    :return: Table of available data per target

    """
    if not isinstance(run_id, str):
        raise ValueError(f"Only single run_id is allowed (str), got {run_id} ({type(run_id)})")

    if exclude_targets is None:
        exclude_targets = []
    if include_targets is None:
        include_targets = []

    is_stored = defaultdict(list)
    for target in self._plugin_class_registry.keys():
        # Skip targets that are not stored
        save_when = self.get_save_when(target)
        if save_when == strax.SaveWhen.NEVER:
            continue

        # Should we include this target or exclude it?
        include_t: ty.Any = []
        exclude_t = False

        for excl in strax.to_str_tuple(exclude_targets):
            # Simple logic, if we match the excluded target, we should
            # should not continue
            if _tag_match(target, excl, pattern_type, False):
                exclude_t = True
                break

        # We can match any of the "incl" targets, keep a list and check
        # of any of the "incl" matches the target.
        for incl in strax.to_str_tuple(include_targets):
            include_t.append(_tag_match(target, incl, pattern_type, False))

        # Convert to simple bool. If no include_targets is specified,
        # all are fine, otherwise check at least one is matching.
        include_t = True if not len(include_t) else any(include_t)

        if include_t and not exclude_t:
            is_stored["target"].append(target)
            is_stored["is_stored"].append(self.is_stored(run_id, target))
    return pd.DataFrame(is_stored)


def _include_exclude_tags(
    dsets,
    include_tags,
    exclude_tags,
    pattern_type,
    ignore_underscore,
):
    if include_tags is not None:
        if not isinstance(include_tags, (str, list, tuple)):
            raise ValueError("include_tags must be a str, list or tuple")
        dsets = dsets[_tags_match(dsets, include_tags, pattern_type, ignore_underscore)]

    if exclude_tags is not None:
        if not isinstance(exclude_tags, (str, list, tuple)):
            raise ValueError("exclude_tags must be a str, list or tuple")
        dsets = dsets[True ^ _tags_match(dsets, exclude_tags, pattern_type, ignore_underscore)]
    return dsets


def _tags_match(dsets, patterns, pattern_type, ignore_underscore):
    result = np.zeros(len(dsets), dtype=bool)

    if isinstance(patterns, str):
        patterns = [patterns]

    for i, tags in enumerate(dsets.tags):
        result[i] = any(
            [
                any(
                    [
                        _tag_match(tag, pattern, pattern_type, ignore_underscore)
                        for tag in tags.split(",")
                        for pattern in patterns
                    ]
                )
            ]
        )

    return result


def _tag_match(tag, pattern, pattern_type, ignore_underscore):
    if ignore_underscore and tag.startswith("_") and not pattern.startswith("_"):
        tag = tag[1:]
    if pattern_type == "fnmatch":
        return fnmatch.fnmatch(tag, pattern)
    elif pattern_type == "re":
        return bool(re.match(pattern, tag))
    raise NotImplementedError


@export
def flatten_run_metadata(md):
    # Flatten the tags field. Note this sets it to an empty string
    # if it does not exist.
    return strax.flatten_dict(
        md, separator=".", keep=[strax.RUN_DEFAULTS_KEY, "sub_run_spec", "tags"]
    )
