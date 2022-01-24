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
    cmt.write('test_db', df)
    # read from the DB
    df2 = cmt.read('test_db')
    # pandas.DataFrame should be identical 
    assert_frame_equal(df, df2)

def test_change_future():
    # add a new value in the future
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test_change_future', df)
    df2 = cmt.read('test_change_future')
    df2.loc[pd.to_datetime(datetime(2029, 12, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [15.0, 13.0, np.nan]
    df2 = df2.sort_index()
    cmt.write('test_change_future', df2)

def test_modify_nan():
    # modify non-physical values (NaN)
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test_modify_nan', df)
    df2 = cmt.read('test_modify_nan')
    df2.loc[pd.to_datetime(datetime(2021, 9, 23, 0, 0, 0, 0, tzinfo=pytz.utc))] = [8.0, 14.0, 14.3]
    df2 = df2.sort_index()
    cmt.write('test_modify_nan', df2)

def test_change_past():
    # modify things in the past
    # fail unless raises a ValueError
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test_change_past', df)
    df2 = cmt.read('test_change_past')
    df2.loc[pd.to_datetime(datetime(2021, 1, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [7.0, 24.0, 14.3]
    df2 = df2.sort_index()
    with unittest.TestCase().assertRaises(ValueError):
         cmt.write('test_change_past', df2)

def test_add_row():
    # add a new row for a new past date, allowed example
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test_add_row', df)
    df2 = cmt.read('test_add_row')
    df2.loc[pd.to_datetime(datetime(2021, 2, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [10.0, np.nan, 14.3]
    df2 = df2.sort_index()
    cmt.write('test_add_row', df2)

def test_add_row_2():
    # add a new row for a new past date, not allowed example
    cmt = corrections()
    df = make_dummy_df()
    cmt.write('test_add_row_2', df)
    df2 = cmt.read('test_add_row_2')
    df2.loc[pd.to_datetime(datetime(2021, 2, 1, 0, 0, 0, 0, tzinfo=pytz.utc))] = [7.0, 24.0, 14.3]
    df2 = df2.sort_index()
    with unittest.TestCase().assertRaises(ValueError):
        cmt.write('test_add_row_2', df2)
