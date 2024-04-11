"""I/O format for corrections using MongoDB.

This module list, reads, writes, among others functionalities to a MongoDB where corrections are
stored.

"""

from datetime import datetime
from datetime import timezone
import logging
import pandas as pd
import pdmongo as pdm
import pymongo
import pytz
import numpy as np

import strax

export, __all__ = strax.exporter()


@export
class CorrectionsInterface:
    """A class to manage corrections that are stored in a MongoDB, corrections are defined as
    pandas.DataFrame with a pandas.DatetimeIndex in UTC, a v1 and online version must be specified,
    online versions are meant for online processing, whereas v1, v2, v3...

    are meant for offline processing. A Global configuration can be set, this means a unique set of
    correction maps.

    """

    def __init__(
        self,
        client=None,
        database_name="corrections",
        host=None,
        username=None,
        password=None,
    ):
        """Start the CorrectionsInterface. To initialize you need either:

            - a pymongo.MongoClient (add as argument client) OR
            - the credentials and url to connect to the pymongo instance
            one wants to be using.
        :param client: pymongo client, a pymongo.MongoClient object
        :param database_name: Database name

        (optional if client is not provided)
        :param host: DB host or IP address e.g. "127.0.0.1"
        :param username: Database username
        :param password: Database password

        """
        # Let's see if someone just provided a pymongo.MongoClient
        if (client is not None) and (host is None and username is None and password is None):
            if not isinstance(client, type(pymongo.MongoClient())):
                raise TypeError(f"{client} is not a pymongo.MongoClient.")
            self.client = client
        # In this case, let's just initialize a new pymongo.MongoClient
        elif (client is None) and (
            host is not None and username is not None and password is not None
        ):
            self.client = pymongo.MongoClient(host=host, username=username, password=password)
        else:
            # Let's not be flexible with our inputs to prevent later
            # misunderstandings because someone thought to be handling a
            # different client than anticipated.
            raise ValueError(
                'Can only init using *either* the "client" or the '
                'combination of "host+username+password", not both'
            )

        self.database_name = database_name

    def list_corrections(self):
        """Smart logic to list all corrections available in the corrections database."""
        database = self.client[self.database_name]
        return [x["name"] for x in database.list_collections()]

    def read_at(self, correction, when, limit=1):
        """Smart logic to read corrections at given time (index), i.e by datetime index.

        :param correction: pandas.DataFrame object name in the DB (str type).
        :param when: when, datetime to read the corrections, e.g. datetime(2020, 8, 12, 21, 4, 32,
            7, tzinfo=pytz.utc)
        :param limit: how many indexes after and before when, i.e. limit=1 will return 1 index
            before and 1 after
        :return: DataFrame as read from the corrections database with time index or None if an empty
            DataFrame is read from the database

        """
        before_df = pdm.read_mongo(
            correction, self.before_date_query(when, limit), self.client[self.database_name]
        )
        after_df = pdm.read_mongo(
            correction, self.after_date_query(when, limit), self.client[self.database_name]
        )

        df = pd.concat([before_df, after_df])

        return self.sort_by_index(df)

    def read(self, correction):
        """Smart logic to read corrections.

        :param correction: pandas.DataFrame object name in the DB (str type).
        :return: DataFrame as read from the corrections database with time index or None if an empty
            DataFrame is read from the database

        """
        df = pdm.read_mongo(correction, [], self.client[self.database_name])

        return self.sort_by_index(df)

    def interpolate(self, what, when, how="interpolate", **kwargs):
        """Interpolate values of a given quantity ('what') of a given correction.

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

        df_new = pd.DataFrame.from_dict({"Time": [when]})

        df_new = df_new.set_index("Time")

        df_combined = pd.concat([df, df_new], sort=False)

        df_combined = df_combined.sort_index()
        if how == "interpolate":
            df_combined = df_combined.interpolate(**kwargs)  # method='linear' is the default
        elif how == "fill":
            df_combined = df_combined.ffill(**kwargs)
        else:
            raise ValueError("Specify an interpolation method, e.g. interpolate or fill")

        return df_combined

    def get_context_config(self, when, global_config="global", global_version="v1"):
        """Global configuration logic.

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
            context_config[correction] = df.loc[df.index == when, version].values[0]

        return context_config

    def write(self, correction, df, required_columns=("ONLINE", "v1")):
        """Smart logic to write corrections to the corrections database.

        :param correction: corrections name (str type)
        :param df: pandas.DataFrame object a DatetimeIndex
        :param required_columns: DataFrame must include two columns online, an ONLINE version and
            OFFLINE version (e.g. v1)

        """
        for req in required_columns:
            if req not in df.columns:
                raise ValueError(f"Must specify {req} in dataframe")
        # Compare against existing data
        logging.info("Reading old values for comparison")
        df_old = self.read(correction)
        if df_old is not None:
            now = datetime.now(tz=timezone.utc)
            new_dates = df.index.difference(df_old.index)
            new_past_dates = new_dates[new_dates < now]
            old_past_dates = df_old.index[df_old.index < now]
            for column in df_old.columns:
                if "ONLINE" in column:
                    # We cannot change ONLINE values in the past
                    if not (
                        df_old.loc[old_past_dates, column] == df.loc[old_past_dates, column]
                    ).all():
                        raise ValueError(
                            f"Existing {column} values must not be changed in the past"
                        )
                    # We can add a new date(row) in the past(ONLINE) as long as
                    # it has the same correction value as for the preceding existing time stamp
                    if not new_past_dates.empty:
                        for new_past_date in new_past_dates:
                            new_value = df.loc[
                                df.index == new_past_date.to_pydatetime(), column
                            ].values[0]
                            preceding_old_value = df_old.loc[
                                df_old.index < new_past_date.to_pydatetime(), column
                            ][-1]
                            if new_value != preceding_old_value:
                                raise ValueError(
                                    f"Adding new past dates to {column} only allowed if correction"
                                    " value not deviating from value for preceding existing time"
                                    " stamp"
                                )
                else:
                    # We can change OFFLINE values in the past only if they are NaN
                    if not (
                        (df_old.loc[old_past_dates, column] == df.loc[old_past_dates, column])
                        | (np.isnan(df_old.loc[old_past_dates, column]))
                    ).all():
                        raise ValueError(f"{column} only NaN values may be updated in the past")
                    # We can add a new date(row) in the past(OFFLINE) only if
                    # it does not affect already potentially processed times,
                    # or if it is the same value as the entire column
                    # (relevant e.g. for indices or constant corrections), or if we only add NaN
                    if not new_past_dates.empty:
                        for new_past_date in new_past_dates:
                            new_value = df.loc[
                                df.index == new_past_date.to_pydatetime(), column
                            ].values[0]
                            if not (np.isnan(new_value) or (df[column] == new_value).all()):
                                preceding_old_value = df_old.loc[
                                    df_old.index < new_past_date.to_pydatetime(), column
                                ][-1]
                                if (
                                    df_old.loc[df_old.index > new_past_date.to_pydatetime(), column]
                                ).empty:
                                    succeeding_old_value = False
                                else:
                                    succeeding_old_value = df_old.loc[
                                        df_old.index > new_past_date.to_pydatetime(), column
                                    ][0]
                                if not np.isnan(
                                    np.array([preceding_old_value, succeeding_old_value])
                                ).any():
                                    raise ValueError(
                                        f"Given new value in {column} not allowed for given past"
                                        " time stamp"
                                    )

        df = df.reset_index()
        logging.info("Writing")

        database = self.client[self.database_name]
        return df.to_mongo(correction, database, if_exists="replace")

    @staticmethod
    def before_date_query(date, limit=1):
        return [
            {
                "$match": {
                    "time": {
                        "$lte": pd.to_datetime(date),
                    }
                }
            },
            {"$sort": {"time": -1}},
            {"$limit": limit},
        ]

    @staticmethod
    def after_date_query(date, limit=1):
        return [
            {
                "$match": {
                    "time": {
                        "$gte": pd.to_datetime(date),
                    }
                }
            },
            {"$sort": {"time": 1}},
            {"$limit": limit},
        ]

    @staticmethod
    def check_timezone(date):
        """Smart logic to check date is given in UTC time zone.

        Raises ValueError if not.
        :param date: date e.g. datetime(2020, 8, 12, 21, 4, 32, 7, tzinfo=pytz.utc)
        :return: the inserted date

        """
        if date.tzinfo == pytz.utc:
            return date
        else:
            raise ValueError(
                f"{date} must be in UTC timezone. Insert datetime object "
                'like "datetime.datetime.now(tz=datetime.timezone.utc)"'
            )

    @staticmethod
    def sort_by_index(df):
        """Smart logic to sort dataframe by index using time column.

        :retrun: df sorted by index(time)

        """
        if df.size == 0:
            return None
        # Delete internal Mongo identifier
        del df["_id"]

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df = df.sort_index()

        return df
