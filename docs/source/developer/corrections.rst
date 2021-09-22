Corrections
========

Overview
---------
Corrections is a centralized interface that allows to store, query, and retrieve information about detector effects (corrections) where this information cab used at the event building process to remove (correct) such effects for a given data type. The information is stored in MongoDB as collection using ``pandas.DataFrame()`` format and with a pandas.DatetimeIndex() this allows track time-dependent information as often detector conditions change over time. Corrections also add the functionality to differentiate between ONLINE and OFFLINE versioning, where ONLINE corrections are used during online processing and, therefore, changes in the past are not allowed, and OFFLINE version meant to be used for re-processing where changes in the past are allowed. Below we explain key features of the corrections class:

*  ``read``: Retrive entire collection as ``pandas.DataFrame()``
*  ``read_at``: Retrieve collection based on a time period (indexes) with limit rows(documents), using indexes greatly reduces the number of documents MongoDB needs to scan, then is a faster method for querying specific information;
*  ``write``: Store (save) entire collection as ``pandas.DataFrame()`` in the DB.
*  ``interpolate``: Often, data is limited in any DB then interpolation is needed when trying to retrieve information at a given time (DateTime). User has the option to use pandas interpolation methods see, e.g.  `link <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html>`_.
  

Finally, a few remarks regarding modifications of collection(``pandas.DataFrame()``). For convention, the user should provide dates(index) in UTC format. In addition, the user has the flexibility to modify or add rows (documents) to any ``pandas.DataFrame()`` (collections) with the only requirement the changes in the past are only for OFFLINE values, for instance, there could be some scenarios where user wants to add a new date (DateTime index) or wants to fill out non-physical values (NaNs) later.
