'''I/O format for corrections using MongoDB

This module list, reads, writes, among others functionalities
to a MongoDB where corrections are stored.
'''

from datetime import datetime
import logging
import pandas as pd
import pdmongo as pdm
import pymongo

import strax

export, __all__ = strax.exporter()


@export
class CorrectionsInterface:
    '''A class to manage corrections that are stored in a MongoDB
    corrections are defined as pandas.DataFrame with a
    pandas.DatetimeIndex, an v1 and online version must be specified,
    online versions are meant for online processing, whereas v1 and
    so on are meant for offline processing. A Global configuration
    is available, meaning a unique set of correction maps.
    '''

    def __init__(self, host='127.0 0.1', username=None, password=None,
                 database_name='corrections'):

        self.host = host
        self.username = username
        self.password = password
        self.client = pymongo.MongoClient(host=self.host,
                                          username=self.username,
                                          password=self.password)
        self.database_name = database_name
        self.database = self.client[self.database_name]

    def list_corrections(self):
        '''Smart logic to list all corrections.
        '''
        return [x['name'] for x in self.database.list_collections()]

    def read(self, correction):
        '''Smart logic to read corrections,
        :param correction: correction's name (str type).
        '''
        df = pdm.read_mongo(correction, [], self.database)

        # No data found
        if df.size == 0:
            return None
        # Delete internal Mongo identifier
        del df['_id']
        return df.set_index('time')

    def interpolate(self, what, when, how='interpolate', **kwargs):
        '''Interpolate values of a given correction
        For information of interpolation methods see,
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        :param what: what do you want to interpolate, what correction(DataFrame)
        :param when: date,  e.g. datetime(2020, 8, 12, 21, 4, 32, 726110)
        :param how: Interpolation method
        '''
        df = what

        df_new = pd.DataFrame.from_dict({'Time': [when]})

        df_new = df_new.set_index('Time')

        df_combined = pd.concat([df, df_new], sort=False)

        df_combined = df_combined.sort_index()
        if how == 'interpolate':
            df_combined = df_combined.interpolate(**kwargs)  # method='linear' is the default
        elif how == 'fill':
            df_combined = df_combined.ffill(**kwards)
        else:
            raise ValueError('Specify an interpolation method, e.g. interpolate or fill')

        return df_combined

    def get_context_config(self, when, global_config='global',
                           global_version='v1'):
        '''Global configuration logic
        :param when: date e.g. datetime(2020, 8, 12, 21, 4, 32, 726110)
        :param global_config: a map of corrections
        :param global_version: global configuration's version
        '''
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
        '''Smart logic to write corrections
        :param correction: corrections' name (str type)
        :param df: pandas DataFrame.
        :required_columns: DataFrame must include a online and v1 columns
        '''
        for req in required_columns:
            if req not in df.columns:
                raise ValueError(f'Muts specify {req} in dataframe')
        # Compare against
        logging.info('Reading old values for comparison')
        df2 = self.read(correction)

        if df2 is not None:
            logging.info('Checking if columns unchanged in past')
            now = datetime.utcnow()
            for column in df2.columns:
                logging.debug(f'Checking {column}')
                if not (df2.loc[df2.index < now, column] ==
                        df.loc[df.index < now, column]).all():
                    raise ValueError(f'{column} changed in past, not allowed')

        df = df.reset_index()
        logging.info('Writing')

        return df.to_mongo(correction, self.database, if_exists='replace')
