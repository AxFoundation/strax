"""Context methods dealing with run scanning and selection"""
import fnmatch
import re
import typing as ty

import numpy as np
import pandas as pd
from tqdm import tqdm

import strax
export, __all__ = strax.exporter()


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
def scan_runs(self: strax.Context,
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
        + ['name', 'number', 'tags', 'mode',
           strax.RUN_DEFAULTS_KEY]
        + list(self.context_config['store_run_fields'])))
    check_available = tuple(set(
        list(strax.to_str_tuple(check_available))
        + list(self.context_config['check_available'])))

    for target in check_available:
        p = self._plugin_class_registry[target]
        if p.save_when < strax.SaveWhen.ALWAYS:
            self.log.warning(f'{p.__name__}-plugin is {str(p.save_when)}. '
                             f'Therefore {target} is most likely not stored!')

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
            doc.setdefault('name', f"{doc['number']:06d}")

            doc.setdefault('mode', '')

            # Convert tags list to a ,separated string
            doc['tags'] = ','.join([t['name']
                                   for t in doc.get('tags', [])])

            # Set a default livetime if we have start and stop
            if ('start' in store_fields
                    and 'end' in store_fields
                    and 'livetime' in store_fields
                    and doc.get('start') is not None
                    and doc.get('end') is not None):
                doc.setdefault('livetime', doc['end'] - doc['start'])

            # Put the strax defaults stuff into a different cache
            if strax.RUN_DEFAULTS_KEY in doc:
                self._run_defaults_cache[doc['name']] = \
                    doc[strax.RUN_DEFAULTS_KEY]
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
            docs = pd.concat([
                docs,
                new_docs[
                    ~np.in1d(new_docs['name'], docs['name'])]],
                sort=False)

    # Rearrange columns
    if (not self.context_config['use_per_run_defaults']
        and strax.RUN_DEFAULTS_KEY in docs.columns):
        del docs[strax.RUN_DEFAULTS_KEY]
    docs = docs[['name'] + [x for x in docs.columns.tolist()
                            if x != 'name']]
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
        self.scan_runs(check_available=strax.to_str_tuple(available))
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
            d_available = np.in1d(self.runs.name.values,
                                  self.list_available(d))
            # Save both in the context and for this selection using
            # available = ('data_type',)
            self.runs[d + '_available'] = d_available
            dsets[d + '_available'] = d_available
    for d in have_available:
        dsets = dsets[dsets[d + '_available']]

    return dsets


@strax.Context.add_method
def define_run(self: strax.Context,
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
        elif not 'run_id' in data:
            raise ValueError(
                "Must provide from_run or data with a run_id column "
                "to define a superrun")
        else:
            df = pd.DataFrame(dict(starts=start, ends=end,
                                   run_id=data['run_id']))
            return self.define_run(
                name,
                {run_id: rs[['start', 'stop']].values.transpose()
                 for run_id, rs in df.groupby('fromrun')})

    if isinstance(data, (list, tuple)):
        # list of runids
        data = strax.to_str_tuple(data)
        return self.define_run(
            name,
            {run_id: 'all' for run_id in data})

    if not isinstance(data, dict):
        raise ValueError(f"Can't define run from {type(data)}")

    # Find start and end time of the new run = earliest start time of other runs
    run_md = dict(start=float('inf'), end=0, livetime=0)
    for _subrunid in data:
        doc = self.run_metadata(_subrunid, ['start', 'end'])
        run_md['start'] = min(run_md['start'], doc['start'])
        run_md['end'] = max(run_md['end'], doc['end'])
        run_md['livetime'] += doc['end'] - doc['start']

    # Superrun names must start with an underscore
    if not name.startswith('_'):
        name = '_' + name

    # Dict mapping run_id: array of time ranges or all
    for sf in self.storage:
        if not sf.readonly and sf.can_define_runs:
            sf.define_run(name, sub_run_spec=data, **run_md)
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


@export
def flatten_run_metadata(md):
    # Flatten the tags field. Note this sets it to an empty string
    # if it does not exist.
    return strax.flatten_dict(
        md,
        separator='.',
        keep=[strax.RUN_DEFAULTS_KEY, 'sub_run_spec', 'tags'])
