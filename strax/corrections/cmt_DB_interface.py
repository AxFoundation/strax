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


CLIENT = pymongo.MongoClient(host='xenon1t-daq.lngs.infn.it',
                               username='corrections',
                               password=os.environ['CORRECTIONS_PASSWORD'])
DATABASE_NAME = 'corrections'
DATABASE = CLIENT[DATABASE_NAME]

@export
def list_corrections():
    """Smart logic to list corrections
    """
    return [x['name'] for x in DATABASE.list_collections() if not 'global' in x['name']]


@export
def read(correction):
    """Smart logic to read corrections
    """ 
    df = pdm.read_mongo(correction, [], DATABASE)
    
    # No data found
    if df.size == 0:
        return None
    # Delete internal Mongo identifier
    del df['_id']

    # Set UTC
    #df['time'] = pd.to_datetime(df['time'], utc=True)
    
    return  df.set_index('time')

@export
def interpolate(what, when, how='interpolate'):
    df = what
    
    df_new = pd.DataFrame.from_dict({'Time' : [when],
                                    # 'This' : [True]
                                    })
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

@export
def get_context_config(when, global_version = 'v1', xenon1t=False):
    """Global configuration logic
    """ 
    if xenon1t:
        df_global = read('global_xenon1t')
    else:
        df_global = read('global')
    
    context_config = {}
    
    for correction, version in df_global.iloc[-1][global_version].items():
        df = read(correction)

        df = interpolate(df, when)
        context_config[correction] = df.loc[df.index == when, version].values[0]
    
    return context_config

@export
def write(correction, df):
    """Smart logic to write corrections
    """
    if 'ONLINE' not in df.columns:
        raise ValueError('Must specify ONLINE column')
    if 'v1' not in df.columns:
        raise ValueError('Must specify v1 column')
       
    # Set UTC
    #df.index = df.index.tz_localize('UTC')
    
    # Compare against
    logging.info('Reading old values for comparison')
    df2 = read(correction)
    
    if df2 is not None:
        logging.info('Checking if columns unchanged in past')

        now = datetime.now()
        for column in df2.columns:
            logging.debug('Checking %s' % column)
            if not (df2.loc[df2.index < now, column] == df.loc[df.index < now, column]).all():
                raise ValueError("%s changed in past, not allowed" % column)

    df = df.reset_index()

    logging.info('Writing')

    return df.to_mongo(correction, DATABASE, if_exists='replace')

@export
def get_time(run_id,xenon1t=False):
    '''Get start time from runsDB
    '''

    collection = pymongo.MongoClient('mongodb://pax:@xenon1t-daq.lngs.infn.it:27017/run',password=os.environ['DB_password'])['xenonnt']['run/runs']
    pipeline =[
            {'$match' : {"number" : run_id, "detector": "tpc", "end": {"$exists": True}}},
            {"$project": {'time' : '$start','number' : 1,'_id' : 0}},
            {"$sort": SON([("time", 1)])}]


    if( xenon1t ):
        collection = pymongo.MongoClient('mongodb://pax:@xenon1t-daq.lngs.infn.it:27017/run',password=os.environ['DB_password'])['run']['runs_new']
        pipeline =[
                {'$match' : {"name" : run_id, "detector": "tpc", "end": {"$exists": True}}},
                {"$project": {'time' : '$start','name' : 1,'_id' : 0}},
                {"$sort": SON([("time", 1)])}]

    time=datetime.now() #to save in datetime format
    for t in collection.aggregate(pipeline):
        time= t['time']

    return time
