"""Context methods dealing with run scanning and selection"""
import fnmatch
import re
import typing as ty

import numpy as np
import pandas as pd
from tqdm import tqdm

import strax


@strax.Context.add_method
def list_available(self, target, **kwargs):
    """Return sorted list of run_id's for which target is available
    """
    # TODO duplicated code with with get_iter
    if len(kwargs):
        # noinspection PyMethodFirstArgAssignment
        self = self.new_context(**kwargs)

    if self.runs is None:
        self.scan_runs()

    keys = set([
        self.key_for(run_id, target)
        for run_id in self.runs['name'].values])

    found = set()
    for sf in self.storage:
        remaining = keys - found
        is_found = sf.find_several(list(remaining), **self._find_options)
        found |= set([k for i, k in enumerate(remaining)
                      if is_found[i]])
    return list(sorted([x.run_id for x in found]))


@strax.Context.add_method
def scan_runs(self,
              check_available=tuple(),
              store_fields=tuple()):
    """Update and return self.runs with runs currently available
    in all storage frontends.
    :param check_available: Check whether these data types are available
    Availability of xxx is stored as a boolean in the xxx_available
    column.
    :param store_fields: Additional fields from run doc to include
    as rows in the dataframe.

    The context options scan_availability and store_run_fields list
    data types and run fields, respectively, that will always be scanned.
    """
    store_fields = tuple(set(
        list(strax.to_str_tuple(store_fields))
        + ['name', 'number', 'tags', 'mode']
        + list(self.context_config['store_run_fields'])))
    check_available = tuple(set(
        list(strax.to_str_tuple(check_available))
        + list(self.context_config['check_available'])))

    docs = None
    for sf in self.storage:
        _temp_docs = []
        for doc in sf._scan_runs(store_fields=store_fields):
            # If there is no number, make one from the name
            if 'number' not in doc:
                if 'name' not in doc:
                    raise ValueError(f"Invalid run doc {doc}, contains "
                                     f"neither name nor number.")
                doc['number'] = int(doc['name'])

            # If there is no name, make one from the number
            doc.setdefault('name', str(doc['number']))

            doc.setdefault('mode', '')

            # Flatten the tags field, if it exists
            doc['tags'] = ','.join([t['name']
                                    for t in doc.get('tags', [])])

            # Flatten the rest of the doc (mainly in case the mode field
            # is something deeply nested)
            doc = strax.flatten_dict(doc, separator='.')

            _temp_docs.append(doc)

        if len(_temp_docs):
            new_docs = pd.DataFrame(_temp_docs)
        else:
            new_docs = pd.DataFrame([], columns=store_fields)

        if docs is None:
            docs = new_docs
        else:
            # Keep only new runs (not found by earlier frontends)
            docs = pd.concat([
                docs,
                new_docs[
                    ~np.in1d(new_docs['name'], docs['name'])]],
                sort=False)

    self.runs = docs

    for d in tqdm(check_available,
                  desc='Checking data availability'):
        self.runs[d + '_available'] = np.in1d(
            self.runs.name.values,
            self.list_available(d))

    return self.runs


@strax.Context.add_method
def select_runs(self, run_mode=None, run_id=None,
                include_tags=None, exclude_tags=None,
                available=tuple(),
                pattern_type='fnmatch', ignore_underscore=True):
    """Return pandas.DataFrame with basic info from runs
    that match selection criteria.
    :param run_mode: Pattern to match run modes (reader.ini.name)
    :param run_id: Pattern to match a run_id or run_ids
    :param available: str or tuple of strs of data types for which data
    must be available according to the runs DB.

    :param include_tags: String or list of strings of patterns
        for required tags
    :param exclude_tags: String / list of strings of patterns
        for forbidden tags.
        Exclusion criteria  have higher priority than inclusion criteria.
    :param pattern_type: Type of pattern matching to use.
        Defaults to 'fnmatch', which means you can use
        unix shell-style wildcards (`?`, `*`).
        The alternative is 're', which means you can use
        full python regular expressions.
    :param ignore_underscore: Ignore the underscore at the start of tags
        (indicating some degree of officialness or automation).

    Examples:
     - `run_selection(include_tags='blinded')`
        select all datasets with a blinded or _blinded tag.
     - `run_selection(include_tags='*blinded')`
        ... with blinded or _blinded, unblinded, blablinded, etc.
     - `run_selection(include_tags=['blinded', 'unblinded'])`
        ... with blinded OR unblinded, but not blablinded.
     - `run_selection(include_tags='blinded',
                      exclude_tags=['bad', 'messy'])`
       select blinded dsatasets that aren't bad or messy
    """
    if self.runs is None:
        self.scan_runs()
    dsets = self.runs.copy()

    if pattern_type not in ('re', 'fnmatch'):
        raise ValueError("Pattern type must be 're' or 'fnmatch'")

    # Filter datasets by run mode and/or name
    for field_name, requested_value in (
            ('name', run_id),
            ('mode', run_mode)):

        if requested_value is None:
            continue

        values = dsets[field_name].values
        mask = np.zeros(len(values), dtype=np.bool_)

        if pattern_type == 'fnmatch':
            for i, x in enumerate(values):
                mask[i] = fnmatch.fnmatch(x, requested_value)
        elif pattern_type == 're':
            for i, x in enumerate(values):
                mask[i] = bool(re.match(requested_value, x))

        dsets = dsets[mask]

    if include_tags is not None:
        dsets = dsets[_tags_match(dsets,
                                  include_tags,
                                  pattern_type,
                                  ignore_underscore)]

    if exclude_tags is not None:
        dsets = dsets[True ^ _tags_match(dsets,
                                         exclude_tags,
                                         pattern_type,
                                         ignore_underscore)]

    have_available = strax.to_str_tuple(available)
    for d in have_available:
        if not d + '_available' in dsets.columns:
            # Get extra availability info from the run db
            self.runs[d + '_available'] = np.in1d(
                self.runs.name.values,
                self.list_available(d))
        dsets = dsets[dsets[d + '_available']]

    return dsets


@strax.Context.add_method
def define_run(self,
               name: str,
               data: ty.Union[np.ndarray, pd.DataFrame, dict],
               from_run: ty.Union[str, None] = None):
    if isinstance(data, (pd.DataFrame, np.ndarray)):
        # Array of events / regions of interest
        start, end = data['time'], strax.endtime(data)
        if from_run is not None:
            return self.define_run(
                name,
                {from_run: np.transpose([start, end])})
        else:
            df = pd.DataFrame(dict(starts=start, ends=end,
                                   run_id=data['run_id']))
            self.define_run(
                name,
                {run_id: rs[['start', 'stop']].values.transpose()
                 for run_id, rs in df.groupby('fromrun')})

    if isinstance(data, (list, tuple)):
        # list of runids
        data = strax.to_str_tuple(data)
        self.define_run(
            name,
            {run_id: 'all' for run_id in data})

    if not isinstance(data, dict):
        raise ValueError("Can't define run from {type(data)}")

    # Dict mapping run_id: array of time ranges or all
    for sf in self.storage:
        if not sf.readonly and sf.can_define_runs:
            sf.define_run(name, data)
            break
    else:
        raise RuntimeError("No storage frontend registered that allows"
                           " run definition")


def _tags_match(dsets, patterns, pattern_type, ignore_underscore):
    result = np.zeros(len(dsets), dtype=np.bool)

    if isinstance(patterns, str):
        patterns = [patterns]

    for i, tags in enumerate(dsets.tags):
        result[i] = any([any([_tag_match(tag, pattern,
                                         pattern_type,
                                         ignore_underscore)
                              for tag in tags.split(',')
                              for pattern in patterns])])

    return result


def _tag_match(tag, pattern, pattern_type, ignore_underscore):
    if ignore_underscore and tag.startswith('_'):
        tag = tag[1:]
    if pattern_type == 'fnmatch':
        return fnmatch.fnmatch(tag, pattern)
    elif pattern_type == 're':
        return bool(re.match(pattern, tag))
    raise NotImplementedError
