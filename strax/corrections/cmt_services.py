import numpy as np
import strax
from .cmt_DB_interface import *

export, __all__ = strax.exporter()

@export
def get_elife(run_id, global_version = 'v1', xenon1t=False):
    """Get electron lifetime correction
    """ 
    when = strax.get_time(run_id, xenon1t)
    if xenon1t:
        df_global = strax.read('global_xenon1t')
        my_correction = 'elife_xenon1t'
    else:
        df_global = strax.read('global')
        my_correction = 'elife'
  
    try:
        for correction, version in df_global.iloc[-1][global_version].items():
            if my_correction == correction:
                df = strax.read(correction)
                df = strax.interpolate(df, when)
    except KeyError:
            raise ValueError(f'Global version {global_version} not found')

    return df.loc[df.index == when, global_version].values[0]

@export
def get_pmt_gains(run_id, global_version = 'v1', xenon1t=False):
    """Get pmt gains
    """ 
    when = strax.get_time(run_id, xenon1t)
    if xenon1t:
        df_global = strax.read('global_xenon1t')
    else:
        df_global = strax.read('global')
    
    # equivalent to 'to_pe' in gains_model
    gains =[] 
    try:
        for correction, version in df_global.iloc[-1][global_version].items():
            if 'pmt' in correction:
                df = strax.read(correction)
                df = strax.interpolate(df, when)
                gains.append(df.loc[df.index == when, version].values[0])
            
            pmt_gains=np.asarray(gains,dtype=np.float32)

    except KeyError:
        raise ValueError(f'Global version {global_version} not found')

    return pmt_gains
