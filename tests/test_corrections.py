import strax
import pandas as pd
import mongomock
import numpy as np
from datetime import datetime
from pandas._testing import assert_frame_equal
import pytz


def test_corrections():
    """
    Test corrections interface
    Use a mongomock client
    testing write and read usingy dummy dataframe
    """
    dummy_client = mongomock.MongoClient()
    cmt = strax.CorrectionsInterface(client=dummy_client)
    
    df = make_dummy_df()
    # write to the DB
    cmt.write('test', df)
    # read from the DB
    df2 = cmt.read('test')
    # pandas.DataFrame should be identical 
    assert_frame_equal(df, df2)

    # add a new row, inserting new date (this checks whether user changes things in the past) 
    df2.loc[pd.to_datetime(datetime(2020, 2, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [10.0, 12.0, 13.0]
    df2 = df2.sort_index()
    cmt.write('test', df2)

    # add a new value in the future
    df3 = cmt.read('test')
    df3.loc[pd.to_datetime(datetime(2022, 12, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [15.0, 13.0, np.nan]
    cmt.write('test', df3)

    # modify nan values
    df4 = cmt.read('test')
    df4.loc[pd.to_datetime(datetime(2021, 9, 23, 0, 0, 0, 0, tzinfo=pytz.utc))] = [8.0, 14.0, 14.3]
    cmt.write('test', df4)


def make_dummy_df():
    """
    Make a dummy pandas.dataframe()
    """
    
    dates = [datetime(2017, 1, 1), datetime(2021, 1, 1), datetime(2021, 9, 23)]

    df = pd.DataFrame({'ONLINE' : [10.,10., 8.],
                       'v1' : [12., 12., 14.],
                       'v2' : [13., 14., np.nan],
                       'time': dates})

    df['time'] = pd.to_datetime(df['time'], utc=True)

    df = df.set_index('time')

    return df
