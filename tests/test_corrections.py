import strax
import pandas as pd
import mongomock
import numpy as np
from datetime import datetime
from pandas._testing import assert_frame_equal
import pytz
import pymongo
import pytest
import unittest

@mongomock.patch()
def corrections():
    dummy_client = pymongo.MongoClient()
    cmt = strax.CorrectionsInterface(client=dummy_client)
    return cmt

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

def test_db():
    cmt = corrections()
    df = make_dummy_df()
    # write to the DB
    cmt.write('test', df)
    # read from the DB
    df2 = cmt.read('test')
    # pandas.DataFrame should be identical 
    assert_frame_equal(df, df2)

def test_change_future():
    # add a new value in the future
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test', df)
    df2 = cmt.read('test')
    df2.loc[pd.to_datetime(datetime(2022, 12, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [15.0, 13.0, np.nan]
    df2 = df2.sort_index()
    cmt.write('test', df2)

def test_modify_nan():
    # modify non-physcal values (nan)
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test', df)
    df2 = cmt.read('test')
    df2.loc[pd.to_datetime(datetime(2021, 9, 23, 0, 0, 0, 0, tzinfo=pytz.utc))] = [8.0, 14.0, 14.3]
    df2 = df2.sort_index()
    cmt.write('test', df2)
    
def test_change_past():
    # modify things in the past
    # fail unless raises the error
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test', df)
    df2 = cmt.read('test')
    df2.loc[pd.to_datetime(datetime(2021, 1, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [7.0, 24.0, 14.3]
    df2 = df2.sort_index()
    with unittest.TestCase().failUnlessRaises(ValueError):
         cmt.write('test', df2)
   
def test_add_row():
    # add a new row in between existen values
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test', df)
    df2 = cmt.read('test')
    # add a new row, inserting new date (this checks whether user changes things in the past) 
    df2.loc[pd.to_datetime(datetime(2020, 2, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [10.0, 12.0, 13.0]
    df2 = df2.sort_index()
    cmt.write('test', df2)
