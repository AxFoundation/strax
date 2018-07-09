import os
import os.path as osp
import warnings

import strax


class DataNotAvailable(Exception):
    pass


class AmbiguousDataRequest(Exception):
    def __init__(self, found, message=''):
        super().__init__(message)
        self.found = found


class DataExistsError(Exception):
    def __init__(self, at, message=''):
        super().__init__(message)
        self.at = at


class DataRegistry:
    """Interface to a data location list, e.g. a runs database
    or a data directory on the file system.
    """

    def find(self, run_id: str, data_type: str, lineage: dict,
             ignore_lineage=tuple(), ignore_config=tuple(),
             ambiguous='warn'):
        """Return (str: backend name, backend-specific key) to get at data
        Raises DataNotAvailable if data does not exist.
        :param run_id: run id string
        :param data_type: data type name
        :param lineage: Lineage dictionary:
        {data_type: (plugin_name, version, {config_option: value, ...}, ...}
        :param ignore_lineage: list/tuple of plugin names for which no
        plugin name, version, or option check is performed.
        :param ignore_config: list/tuple of configuration options for which no
        check is performed.
        :param ambiguous: Behaviour if multiple matching data entries are
        found:
         - 'error': Raise AmbigousDataRequest.
         - 'warn': warn with AmbiguousDataDescription. Return first match.
         - 'ignore': do nothing. Return first match.
        """
        message = (
            f"\nRequested lineage: {lineage}."
            f"\nIgnoring versions for: {ignore_lineage}."
            f"\nIgnoring config options for: {ignore_lineage}.")
        try:
            self._find(run_id, data_type, lineage,
                       ignore_lineage, ignore_config)
        except DataNotAvailable:
            raise DataNotAvailable(
                f"{data_type} for {run_id} not available." + message)
        except AmbiguousDataRequest as e:
            found = e.found
            message = (f"Found {len(found)} data entries for {run_id},"
                       f" {data_type}: {found}." + message)
            if ambiguous == 'ignore':
                pass
            elif ambiguous == 'warn':
                warnings.warn(message)
            else:
                if ambiguous != 'error':
                    print(f"Invalid ambiguous argument {ambiguous} "
                          "treated as 'error'.")
                raise AmbiguousDataRequest(found=found, message=message)

    def start_saving(self, run_id: str, data_type: str, lineage: dict):
        """Start saving data. Return (str: backend name, backend-specific key)
        to be used for creating a saver.
        :param run_id: run id string
        :param data_type: data type name
        :param lineage: Lineage dictionary:
        {data_type: (plugin_name, version, {config_option: value, ...}, ...}
        raises DataExistsError error if data already available with this
        exact lineage.
        raises CannotWriteError if desired backend is readonly
        """
        try:
            self._start_saving(run_id, data_type, lineage)
        except DataExistsError as e:
            raise DataExistsError(
                at=e.at,
                message=f"Already stored data for {run_id}, {data_type} with "
                        f"lineage {lineage} at {e.at}.")
        self.start_saving(run_id, data_type, lineage)

    def finish_saving(self, run_id: str, data_type: str, lineage: dict):
        """Indicate saving of data is completed succesfully.
        :param run_id: run id string
        :param data_type: data type name
        :param lineage: Lineage dictionary:
        {data_type: (plugin_name, version, {config_option: value, ...}, ...}
        """
        pass

    def _start_saving(self, run_id, data_type, lineage):
        raise NotImplementedError

    def _find(self, run_id, data_type, lineage,
              ignore_versions, ignore_config):
        raise NotImplementedError

    def matches(self, lineage: dict, desired_lineage: dict,
                ignore_lineage: tuple, ignore_config: tuple):
        """Return if lineage matches desired_lineage given ignore options
        """
        if not (ignore_lineage or ignore_config):
            return lineage == desired_lineage
        args = [ignore_lineage, ignore_config]
        return (
            self._filter_lineage(lineage, *args)
            == self._filter_lineage(desired_lineage, *args))

    @staticmethod
    def _filter_lineage(lineage, ignore_lineage, ignore_config):
        return {data_type: (v[0],
                            v[1],
                            {option_name: b
                             for option_name, b in v[2].items()
                             if option_name not in ignore_config})
                for data_type, v in lineage.items()
                if data_type not in ignore_lineage}


class DataDirectory(DataRegistry):
    """Simplest registry: single directory with FileStore data
    in subdirectories.
    """
    def __init__(self, path='.'):
        self.path = path
        self.filestore = strax.FileStore(path)

    def _find(self, run_id, data_type, lineage,
              ignore_versions, ignore_config):
        # Check for exact match
        dirname = self.dirname(run_id, data_type, lineage)
        if osp.exists(dirname):
            return self.backend_key(dirname)
        if not ignore_versions and not ignore_config:
            raise DataNotAvailable

        # Check metadata of all potentially matching data dirs for match
        for dirname in os.listdir(self.path):
            if not osp.isdir(dirname):
                continue
            if not osp.exists(osp.join(dirname, 'metadata.json')):
                continue
            # TODO: check for broken data
            metadata = self.filestore.get_meta(dirname)
            if self.matches(metadata['lineage'], lineage,
                            ignore_versions, ignore_config):
                return self.backend_key(dirname)
        raise DataNotAvailable

    def dirname(self, run_id, data_type, lineage):
        return osp.join(self.path,
                        '_'.join([run_id, data_type,
                                  strax.deterministic_hash(lineage)]))

    def backend_key(self, dirname):
        return self.filestore.__class__.__name__, dirname

    def _start_saving(self, run_id, data_type, lineage):
        dirname = self.dirname(run_id, data_type, lineage)
        if osp.exists(dirname):
            raise DataExistsError(at=dirname)
        return self.backend_key(dirname)
