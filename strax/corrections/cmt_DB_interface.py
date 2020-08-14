from datetime import datetime
import logging
import os
from bson.son import SON
import pandas as pd
import pdmongo as pdm
import pymongo
import numpy as np

import strax

export, __all__ = strax.exporter()

'''In any given experiment applying corrections help us to remove detectors effects, 
therefore, we define a class to manage corrections that are housed in a 
mongoDB, where the manager can read, write among others functionalities
'''
@export
class Corrections_DB_interface:

    def __init__(self,host=None,username=None,password=None,database_name=None):

        self.host = host
        self.username = username
        self.password = password
        self.client = pymongo.MongoClient(host=self.host,username=self.username,password=self.password)
        self.database_name= database_name 
        self.database=self.client[self.database_name]

    def list_corrections(self):
        '''Smart logic to list corrections.
        '''
        return [x['name'] for x in self.database.list_collections() if not 'global' in x['name']]


    def read(self, correction):
        '''Smart logic to read corrections, 
        where correction is the name of the DataFrame.
        ''' 
        df = pdm.read_mongo(correction, [], self.database)
    
        # No data found
        if df.size == 0:
            return None
        # Delete internal Mongo identifier
        del df['_id']
    
        return  df.set_index('time')

    def interpolate(self, what, when, how='interpolate'):
        '''Interpolate values of a given correction(what=DataFrame)
        given a time(when). For information of interpolation methods 
        see, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
        '''
        df = what
    
        df_new = pd.DataFrame.from_dict({'Time' : [when]})

        df_new = df_new.set_index('Time')
    
        df_combined = pd.concat([df,
                             df_new],
                            sort=False)
        df_combined = df_combined.sort_index()
        if how == 'interpolate':
            df_combined = df_combined.interpolate(method='linear')
        elif how == 'fill':
            df_combined = df_combined.ffill()
        else:
            raise ValueError()
        
        return df_combined

    def get_context_config(self, when, global_version = 'v1', xenon1t=False):
        '''Global configuration logic.
        ''' 
        if xenon1t:
            df_global = self.read('global_xenon1t')
        else:
            df_global = self.read(self,'global')
    
        context_config = {}
    
        for correction, version in df_global.iloc[-1][global_version].items():
            df = Corrections_DB_interface.read(self,correction)

            df = Corrections_DB_interface.interpolate(self,df, when)
            context_config[correction] = df.loc[df.index == when, version].values[0]
    
        return context_config

    def write(self, correction, df):
        '''Smart logic to write corrections
        where, corrections is the name of the correction(str type) and df is pandas DataFrame.
        '''

        if 'ONLINE' not in df.columns:
            raise ValueError('Must specify ONLINE column')
        if 'v1' not in df.columns:
            raise ValueError('Must specify v1 column')
       
        # Compare against
        logging.info('Reading old values for comparison')
        df2 = Corrections_DB_interface.read(self,correction)
    
        if df2 is not None:
            logging.info('Checking if columns unchanged in past')

            now = datetime.utcnow()
            for column in df2.columns:
                logging.debug(f'Checking {column}')
                if not (df2.loc[df2.index < now, column] == df.loc[df.index < now, column]).all():
                    raise ValueError(f'{column} changed in past, not allowed')

        df = df.reset_index()

        logging.info('Writing')

        return df.to_mongo(correction, self.database, if_exists='replace')
