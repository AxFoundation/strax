"""I/O format for corrections using MongoDB

This module list, reads, writes, among others functionalities
to a MongoDB where corrections are stored.
"""

from datetime import datetime
from datetime import timezone
import logging
import pandas as pd
import pdmongo as pdm
import pymongo
import pytz

import strax

export, __all__ = strax.exporter()


@export
class CorrectionsInterface:
    """
    A class to manage corrections that are stored in a MongoDB,
    corrections are defined as pandas.DataFrame with a
    pandas.DatetimeIndex in UTC, a v1 and online version must be specified,
    online versions are meant for online processing, whereas v1, v2, v3...
    are meant for offline processing. A Global configuration can be set,
    this means a unique set of correction maps.
    """

    def __init__(self, host='127.0.0.1', username=None, password=None,
                 database_name='corrections'):
        """
        :param host: DB host
        :param username: DB username
        :param password: DB password
        :param database_name: DB name
        """

        self.host = host
        self.username = username
        self.password = password
        self.client = pymongo.MongoClient(host=self.host,
                                          username=self.username,
                                          password=self.password)
        self.database_name = database_name
        self.database = self.client[self.database_name]

    def list_corrections(self):
        """
        Smart logic to list all corrections available in the corrections database
        """
        return [x['name'] for x in self.database.list_collections()]

    def read(self, correction):
        """Smart logic to read corrections,
        :param correction: pandas.DataFrame object name in the DB (str type).
        :return: DataFrame as read from the corrections database with time
        index or None if an empty DataFrame is read from the database
        """
        df = pdm.read_mongo(correction, [], self.database)

        # No data found
        if df.size == 0:
            return None
        # Delete internal Mongo identifier
        del df['_id']

        df['time'] = pd.to_datetime(df['time'], utc=True)

        return df.set_index('time')

    def interpolate(self, what, when, how='interpolate', **kwargs):
        """
        Interpolate values of a given quantity ('what') of a given correction.
        For information of interpolation methods see:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        :param what: what do you want to interpolate, what correction(DataFrame)
        :param when: date, e.g. datetime(2020, 8, 12, 21, 4, 32, 7, tzinfo=pytz.utc)
        :param how: Interpolation method, can be either 'interpolate' or 'fill'
        :param kwargs: are forward to the interpolation
        :return: DataFrame of the correction with the interpolated time ('when')
        """
        # Check the user input
        self.check_timezone(when)

        df = what

        df_new = pd.DataFrame.from_dict({'Time': [when]})

        df_new = df_new.set_index('Time')

        df_combined = pd.concat([df, df_new], sort=False)

        df_combined = df_combined.sort_index()
        if how == 'interpolate':
            df_combined = df_combined.interpolate(**kwargs)  # method='linear' is the default
        elif how == 'fill':
            df_combined = df_combined.ffill(**kwargs)
        else:
            raise ValueError('Specify an interpolation method, e.g. interpolate or fill')

        return df_combined

    def get_context_config(self, when, global_config='global',
                           global_version='v1'):
        """
        Global configuration logic
        :param when: date e.g. datetime(2020, 8, 12, 21, 4, 32, 7, tzinfo=pytz.utc)
        :param global_config: a map of corrections
        :param global_version: global configuration's version
        :return: configuration (type: dict)
        """
        # Check the user input
        self.check_timezone(when)
        df_global = self.read(global_config)

        context_config = {}
        # loop over corrections and versions to get a global context
        for correction, version in df_global.iloc[-1][global_version].items():
            df = self.read(correction)
            df = self.interpolate(df, when)
            context_config[correction] = df.loc[df.index == when,
                                                version].values[0]

        return context_config

    def write(self, correction, df, required_columns=('ONLINE', 'v1')):
        """
        Smart logic to write corrections to the corrections database.
        :param correction: corrections is a pandas.DataFrame object,
        corrections name (str type)
        :param df: corrections is a pandas.DataFrame object a DatetimeIndex
        :param required_columns: DataFrame must include an online and v1 columns
        """
        for req in required_columns:
            if req not in df.columns:
                raise ValueError(f'Must specify {req} in dataframe')
        # Compare against
        logging.info('Reading old values for comparison')
        df2 = self.read(correction)

        if df2 is not None:
            logging.info('Checking if columns unchanged in past')
            now = datetime.now(tz=timezone.utc)
            for column in df2.columns:
                logging.debug(f'Checking {column}')
                if not (df2.loc[df2.index < now, column] ==
                        df.loc[df.index < now, column]).all():
                    raise ValueError(f'{column} changed in past, not allowed')

        df = df.reset_index()
        logging.info('Writing')

        return df.to_mongo(correction, self.database, if_exists='replace')

    @staticmethod
    def check_timezone(date):
        """
        Smart logic to check date is given in UTC time zone. Raises ValueError
        if not.
        :param date: date e.g. datetime(2020, 8, 12, 21, 4, 32, 7, tzinfo=pytz.utc)
        :return: the inserted date
        """
        if date.tzinfo == pytz.utc:
            return date
        else:
            raise ValueError(f'{date} must be in UTC timezone. Insert datetime object '
                             f'like "datetime.datetime.now(tz=datetime.timezone.utc)"')
