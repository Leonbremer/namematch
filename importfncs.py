"""
Project:    MPhil thesis
Code:       Import functions (importfncs.py)
@author:    Leon Bremer
@date:      July 8, 2020

Import functions for the downloaded databases
    - PATSTAT
    - Amadeus Financials
    - Amadeus Subsidiaries
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from datetime import datetime as dt
from itertools import repeat

import parameters as pars


def import_ama(table, cols):
    """
    Imports the requested columns (cols) from the Amadeus Financials or
    Amadeus Subsidiaries database.
    
    Arguments:
        table: fin or sub, for the Financials or Subsidiaries database.
        cols: columns
        fnargs: tuple of functions and arguments to those functions
    
    return: 
        df: dataframe
    """
    ### Parameters
    tablepos = ['fin', 'sub']
    if table.lower()[:3] not in tablepos:
        print('table must be in', tablepos)
    dtypes = {k:v for (k,v) in pars.ama.dtypes[table].items() if k in cols}
    if table.lower()[:3]=='sub':
        cols_to_float = ['SUBS_EMPL','SUBS_TOAS','SUBS_OPRE']
        dtypes = {k:v for (k,v) in dtypes.items() if k not in cols_to_float}
        
    tStart = dt.now()
    print('+++START......importing Amadeus', table)
    
    ### Import data
    df = pd.read_csv(
        pars.paths.data.raw_ama(table),
        usecols = cols,
        encoding = pars.ama.encoding,
        dtype = dtypes)
    
    if table.lower()[:3]=='sub':
        df.loc[:,cols_to_float] = df[cols_to_float].apply(pd.to_numeric,
                                                          errors='coerce')
    
    df.reset_index(drop = True, inplace = True)
    
    print('---FINISH.....importing Amadeus', table, ' in',
          dt.now() - tStart)
    return df


def import_ps_file(file, cols, dtypes=None, filter_sector=False):
    """
    Use pandas read_csv() in the multiprocessing setting for the PATSTAT
    databases.

    Parameters
    ----------
    file : str
        File path to csv file to be read.
    cols : str list
        List of column names to be passed on to read_csv.
    dtypes : dict or str, optional
        Data types to be used for the reading in of the CSVs. The
        default is None.

    Returns
    -------
    df : Pandas DataFrame
        Dataframe from the read in CSV.

    """
    df = pd.read_csv(file, sep=';', usecols=cols, dtype=dtypes)
    
    # Drop individuals (incl. NA which seem unique to individuals)
    if filter_sector:
        df = df[~df['psn_sector'].isin(['INDIVIDUAL', np.NaN])]
    
    return df


def import_ps_folder(table, cols, filter_sector=False):
    """
    Import a table from PATSTAT. Requested columns can be specified
    with cols.
    
    Dtype parameters and folder paths are specfied in parameters.py.
    
    Arguments:
        table: PATSTAT table number to be downloaded
        cols: Requested columns
        fnargs: tuple of functions and arguments to those functions
    
    return:
        df: dataframe of PATSTAT table
    """
    ### Parameters
    iCores = os.cpu_count()
    folder = pars.paths.data.raw_ps_tls(table)
    files = [f for f in os.listdir(folder) if f.startswith('resulttable')]
    file_paths = [folder + f for f in files]
    ps_dtypes = pars.ps.dtypes[str(table)]# all PS dtypes
    dtypes = {k: v for (k, v) in ps_dtypes.items() if k in cols}
    dtypes_noncat = {k: v for (k, v) in dtypes.items() if v!='category'}
    dtypes_cat = {k: v for (k, v) in dtypes.items() if v=='category'}
    
    tStart = dt.now()
    print('+++START......importing PATSTAT folder')
    
    ### Import data
    args = list(zip(file_paths,
                    repeat(cols),
                    repeat(dtypes_noncat),
                    repeat(filter_sector)))
    with mp.Pool(processes=iCores) as pool:
        dfs = pool.starmap(import_ps_file,args)
    
    df = pd.concat(dfs,axis=0)
    df = df.astype(dtypes_cat)
    df.reset_index(drop=True,inplace=True)
    
    print('---FINISH.....importing PATSTAT folder in', dt.now() - tStart)
    return df