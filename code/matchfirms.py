"""
Project:    Fuzzy name matching (namematch)
Code:       Match firm names (matchfirms.py)
@author:    Leon Bremer
@date:      Sept 14, 2023

The Amadeus data has consistent and valid firm identifiers, PATSTAT does
not have consistent applicant identifiers.

The matching procedure takes three steps:
    1. Cleaning the firm names.
    2. Vectorizing the cleaned names and calculating the cosine between
    each vector. A maximum angle determines candidate matches.
    3. A sample of candidates are manually labelled and a machine 
    learning algorithm (probit) is fitted and used to determine the 
    matching status of all non-labelled (out-of-sample) candidates.
    4. Disambiguation cleans the links by removing duplicate links.
    Either links each applicant to one firm or creates firm groups using
    community detection to avoid duplicate links.

To reduce memory and CPU usage consider:
    1. Using sparse matrices, especially when data contains many zeros,
    which is the case when vectorizing the firm names.
    2. Converting data types to smaller formats, e.g. float16 or
    float32, trading off storage vs precision. (Note that sparse 
    matrices do not support np.float16. When using float16, some 
    operations will return errors, and multiplication of two float16 
    sparse matrices results in a float32 sparse matrix. See
    https://github.com/scipy/scipy/issues/8903).
    3. Do not save all results and use generators when possible.
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as scsp
import copy
import os
import random
import statsmodels.api as smf
from itertools import combinations
import networkx as nx

from datamanip.matchfirms import preproc
import datamanip.helpers.importfncs as imp
import parameters as pars

random.seed(12)

# ============================================ FOR TESTING PURPOSES ONLY

def get_samples():
    """
    Randomly draw a sample from the three databases (PATSTAT, Amadeus
    Financials, Amadeus Subsidiaries). For testing purposes only.
    """
    path = pars.paths.data.proc + 'firmnames/'
    seed = 12
    f = 0.05
    databases = ['names_ps_tls206_5.gz',
                 'names_amafin5.gz',
                 'names_amasub5.gz']
    
    l = []
    print('Reading in', str(len(databases)), 'databases.')
    for db in databases:
        tStart = dt.now()
        print('>', db, end='')
        
        df = pd.read_pickle(path+db)
        df = df.sample(frac=f, random_state=seed, axis='index')
        
        print(' - Done in', dt.now()-tStart, '.')
        
        l += [df]
    return tuple(l)


# Create samples and store them in pkl files
if False:
    (df_ps, df_amafin, df_amasub) = get_samples()
    df_ps.to_pickle(pars.paths.data.proc + 'firmnames/' + 'ps_sample.gz')
    df_amafin.to_pickle(pars.paths.data.proc + 'firmnames/'
                        + 'amafin_sample.gz')
    df_amasub.to_pickle(pars.paths.data.proc + 'firmnames/'
                        + 'amasub_sample.gz')

# Load in earlier created samples
if False:
    df_ps = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                           + 'ps_sample.gz')
    df_amafin = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                               + 'amafin_sample.gz')
    df_amasub = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                               + 'amasub_sample.gz')

# Load in all data
if False:
    df_ps = pd.read_pickle(pars.paths.data.proc + 'firmnames/' +
                               'names_ps_tls206_5.gz')
    df_amafin = pd.read_pickle(pars.paths.data.proc + 'firmnames/' +
                                   'names_amafin5.gz')
    df_amasub = pd.read_pickle(pars.paths.data.proc + 'firmnames/' +
                                   'names_amasub5.gz')


# ======================================================= PRE-PROCESSING

def proc(df,cols):
    """
    Preprocess the raw string data. This function is not very flexible
    and should itself be adjusted if other processing is desired.
    """
    ### Parameters
    df_proc = copy.deepcopy(df)
    # Procedures and their arguments
    procs = [preproc.delete_punc,
             preproc.replace_legal,# needs spaces, no punc, ignores case
             preproc.to_lower,
             preproc.replace_german,# needs lowercase
             preproc.delete_punc,# needs lowercase
             preproc.replace_accented,]
    procs_kwargs = [
        {'punc':True, 'quotes': True, 'spaces': False, 'misc': False},
        {},
        {},
        {},
        {'punc':False, 'quotes': False, 'spaces': True, 'misc': True},
        {},]
    # Processing
    for proc, kwa in zip(procs, procs_kwargs):
        df_proc = preproc.proc_multicol(df_proc,cols,proc,**kwa)
        print('.',end='')
    print('')
    return df_proc


# =========================================== COSINE SIMILARITY MATCHING

# Chunk generator
def chunker(matrix,rows):
    """
    Chunk a matrix into pieces along the rows dimension.
    """
    start = 0
    last = matrix.shape[0]
    while start<last:
        yield (start,matrix[start:start+rows,:])
        start+=rows

def cossim(mat1,s,mat2,theta):
    """
    Calculate cosines using the Cosine Similarity and store indexes.
    """    
    # Cosine matrix
    cs = mat1 @ mat2.T
    nz = cs.nnz
    # Indices where threshold is passed
    (i,j) = (cs>math.cos(theta)).nonzero()
    if i.shape[0]>0:
        v = np.arccos(np.minimum(np.ravel(cs[i,j]),1))
        i += s
    else:
        v = np.array([],dtype=np.float32)
    return (i,j,v,nz)


# ============================================================== ML part

def precision_recall(truth,p,p_cutoff):
    """
    Calculates precision and recall of predicted probability p, when
    compared to the truth value and considering cutoff p_cutoff.
    """
    p_bool = p>p_cutoff
    precision = sum(np.logical_and(p_bool,truth))/sum(p_bool)
    recall = sum(np.logical_and(p_bool,truth))/sum(truth)
    f_score = (2*(precision*recall))/(precision+recall)
    return (precision,recall,f_score)

def prep_data_reg():
    # Parameters
    files = os.listdir(pars.paths.data.proc+'firmnames')
    files = [pars.paths.data.proc+'firmnames/'+f 
             for f in files if (f.startswith('matching_results_')
                                and f.endswith('000.xlsx'))]
    dtypes_labelled = {'link_id':np.uint32,
                       'same_start':bool,
                       'theta':np.float64,
                       'name_train':object,
                       'name_match':object,
                       'pass':object,
                       'randomrank':np.uint32}
    # Flags for often-wrong matches
    flags = [
        'zakrytoe','aktsionernoe','obshchestvo',
        'otkrytoe',
        'zaklady','wlokien','chemicznych',
        'przedsiebiorstwo',
        'handlowe',
        'nauchno','proizvodstvennaya','proizvodstvennoe','predpriyatie'
        ]
    flags_alt =[
        'societe','societa'
        ]
    # Data
    df_matches_info = pd.read_pickle(pars.paths.data.proc+'firmnames/'+
                                     'matching_results_all.gz')
    dfs_matches_labelled = [pd.read_excel(f,dtype=dtypes_labelled)
                            for f in files]
    df_matches_labelled = pd.concat(dfs_matches_labelled,axis=0)
    ### Build df with regression variables
    # Keep names not from excel loaded file. Excel might change them
    # (e.g. when name starts with =, excel escapes it with =@, but
    # reading it results in NA)
    df_reg = pd.merge(
        left=df_matches_labelled[['link_id','pass','randomrank']],
        right=df_matches_info[
            ['link_id','name_train','name_match','name_train_proc',
             'name_match_proc','m_train','m_match','same_start','theta']],
        on='link_id',
        how='left')
    ### Add regression variables
    # First two characters in other name
    df_reg.loc[:,'train2inmatch'] = [t in m for (t,m) in zip(
        df_reg['name_train_proc'].str[:2],df_reg['name_match_proc'])]
    df_reg.loc[:,'match2intrain'] = [m in t for (m,t) in zip(
        df_reg['name_match_proc'].str[:2],df_reg['name_train_proc'])]
    df_reg.loc[:,'train2inmatch_int'] = df_reg['train2inmatch'].replace(
        {True: 1, False: 0})
    df_reg.loc[:,'match2intrain_int'] = df_reg['match2intrain'].replace(
        {True: 1, False: 0})
    # Same start as integer
    df_reg.loc[:,'same_start_int'] = df_reg['same_start'].replace(
        {True: 1, False: 0})
    # Flagged
    df_reg.loc[:,'flag'] = df_reg['name_train_proc'].str.contains(
        '|'.join(flags)) | df_reg['name_match_proc'].str.contains(
            '|'.join(flags))
    df_reg.loc[:,'flag_int'] = df_reg['flag'].replace(
        {True: 1, False: 0})
    df_reg.loc[:,'flag_alt'] = df_reg['name_train_proc'].str.contains(
        '|'.join(flags_alt)) & df_reg['name_match_proc'].str.contains(
            '|'.join(flags_alt))
    df_reg.loc[:,'flag_alt_int'] = df_reg['flag_alt'].replace(
        {True: 1, False: 0})
    # Data source
    df_reg.loc[:,'amafin'] = df_reg['m_train']=='NAME'
    df_reg.loc[:,'amafin_int'] = df_reg['amafin'].replace(
        {True: 1, False: 0})
    # Occurences
    df_count_train = df_reg['name_train_proc'].value_counts().rename(
        'train_proc_count')
    df_count_match = df_reg['name_match_proc'].value_counts().rename(
        'match_proc_count')
    df_reg = pd.merge(
        left=df_reg,
        right=df_count_train,
        left_on='name_train_proc',
        right_index=True)
    df_reg = pd.merge(
        left=df_reg,
        right=df_count_match,
        left_on='name_match_proc',
        right_index=True)
    df_reg.loc[:,'train_proc_count_log'] = np.log(df_reg['train_proc_count'])
    df_reg.loc[:,'match_proc_count_log'] = np.log(df_reg['match_proc_count'])
    # Constant
    df_reg.loc[:,'constant'] = 1
    # Interactions
    df_reg.loc[:,'theta_train2inmatch'] = (
        df_reg['train2inmatch']*df_reg['theta'])
    df_reg.loc[:,'theta_match2intrain'] = (
        df_reg['match2intrain']*df_reg['theta'])
    df_reg.loc[:,'theta_same_start'] = (
        df_reg['theta']*df_reg['same_start_int'])
    df_reg.loc[:,'theta_amafin'] = df_reg['amafin_int']*df_reg['theta']
    df_reg.loc[:,'same_start_amafin'] = (
        df_reg['amafin_int']*df_reg['same_start_int'])
    df_reg.loc[:,'theta_match_count_log'] = df_reg['theta'] * df_reg[
        'match_proc_count_log']
    return df_reg

def prep_data_lab(df):
    """
    Add a column with info on which rows are for fitting, testing and
    out-of-sample prediction.
    """
    # Parameters
    randrank = 1_000
    last_link_ids_lower = 1_100_000
    randrank_last = 453
    train_share = 2/3
    # Randomly (manually) labelled
    df_lab = copy.deepcopy(df)
    df_lab.loc[:,'randomlabelled'] = (
        ((df_lab['link_id']<last_link_ids_lower) &
         (df_lab['randomrank']<=randrank)) |
        ((df_lab['link_id']>=last_link_ids_lower) &
         (df_lab['randomrank']<=randrank_last))
    )
    df_lab.loc[:,'truematch'] = df_lab['pass']==True
    # Select only the non-exact matches
    df_lab.loc[:,'nonexact'] = df_lab['theta']>0.01
    # Split in training and testing
    df_lab_ne = df_lab[df_lab['nonexact'] & df_lab['randomlabelled']]
    n_train = math.floor(train_share*len(df_lab_ne))
    links_train = df_lab_ne.sample(n=n_train,axis=0,replace=False,
                                   random_state=12)['link_id']
    links_check = df_lab_ne[~df_lab_ne['link_id'].isin(links_train)]['link_id']
    links_oos = df_lab[
        ~df_lab['link_id'].isin(df_lab_ne['link_id'])]['link_id']
    df_lab.loc[:,'mlpurpose'] = 'UNKNOWN'
    df_lab.loc[df_lab['link_id'].isin(links_oos),'mlpurpose'] = 'oos'
    df_lab.loc[df_lab['link_id'].isin(links_train),'mlpurpose'] = 'train'
    df_lab.loc[df_lab['link_id'].isin(links_check),'mlpurpose'] = 'check'
    return df_lab

def fit_predict_match(df,predictors,buffer_model=None,buffer_predict=None):
    """
    Fits a probit model on a training set, tests it against a test set,
    and predicts the status on an out-of-sample set.
    
    Arguments:
        df: Dataframe for the regression. Note that certain columns have
        to be present with a specific name and format.
        predictors: List of variable (column) names to be used in the
        regresssion.
        buffer_model: Location to save fitted probit model.
        buffer_predict: Location to save outcome df.
    
    Return:
        probit_fit: The fitted probit model.
        df_predict: Dataframe with probit predictions and link_id.
    """
    ### Parameters
    cols_ml_info = ['link_id','randomlabelled','truematch','nonexact',
                    'mlpurpose']
    ### Test df format
    if not (set(cols_ml_info)
            <= set(df.columns)):
        raise ValueError('df does not contain expected columns.')
    ### Data prep
    df_train = df[df['mlpurpose']=='train']
    ### Probit model
    # Fit on training sample
    probit_model = smf.Probit(
        endog=df_train['truematch'],
        exog=df_train[predictors])
    probit_fit = probit_model.fit()
    # Predict for all observations
    probit_predict = probit_model.predict(
        params=probit_fit.params,
        exog=df[predictors])
    df_predict = df[cols_ml_info].assign(prediction=probit_predict)
    ### Save outcomes
    if buffer_model is not None:
        probit_fit.save(buffer_model)
    if buffer_predict is not None:
        df_predict.to_pickle(buffer_predict)
    return (probit_fit,df_predict)

def match_decision(update=False):
    """
    Fit a probit model on a labelled sample and predict true matching
    status of all out-of-sample matching candidates.
    """
    ### Parameters
    # Predictors per model
    predictors_full = [
        'theta',#'theta30',
        'same_start_int','theta_same_start',
        'match2intrain_int','train2inmatch_int',
        # 'theta_train2inmatch','theta_match2intrain',
        'theta_match_count_log',
        'amafin_int','theta_amafin','same_start_amafin',
        'flag_int','flag_alt_int',#'theta_high_flag_alt',#'theta_flag',
        'constant']
    predictors_sep = [
        'theta',#'theta30',
        'same_start_int','theta_same_start',
        'match2intrain_int','train2inmatch_int',
        # 'theta_train2inmatch','theta_match2intrain',
        'theta_match_count_log',
        'flag_int','flag_alt_int',#'theta_high_flag_alt',#'theta_flag',
        'constant']
    ### Data
    # df with regression variables
    buffer_df_reg = (pars.paths.data.proc+'firmnames/'
                     +'dataframe_match_candidates_reg.gz')
    if os.path.exists(buffer_df_reg) and not update:
        df_reg = pd.read_pickle(buffer_df_reg)
    else:
        df_reg = prep_data_reg()
        df_reg = prep_data_lab(df_reg)
        df_reg.to_pickle(buffer_df_reg)
    ### Fit probit models
    fit_kwargs = [
        {'df':df_reg,
         'predictors':predictors_full,
         'buffer_model':(pars.paths.data.proc+'firmnames/'
                         +'probit_fit_full.pkl'),
         'buffer_predict':(pars.paths.data.proc+'firmnames/'
                           +'dataframe_probit_prediction_full.gz')},
        {'df':df_reg[df_reg['m_train']=='NAME'],
         'predictors':predictors_sep,
         'buffer_model':(pars.paths.data.proc+'firmnames/'
                         +'probit_fit_amafin.pkl'),
         'buffer_predict':(pars.paths.data.proc+'firmnames/'
                           +'dataframe_probit_prediction_amafin.gz')},
        {'df':df_reg[df_reg['m_train']=='SUBS_NAME'],
         'predictors':predictors_sep,
         'buffer_model':(pars.paths.data.proc+'firmnames/'
                         +'probit_fit_amasub.pkl'),
         'buffer_predict':(pars.paths.data.proc+'firmnames/'
                           +'dataframe_probit_prediction_amasub.gz')}]
    probit_results = [fit_predict_match(**kwargs) for kwargs in fit_kwargs]
    ### Determine F1-scores
    thresholds = np.linspace(.05,.95,181)
    probit_performance = [
        pd.DataFrame([precision_recall(df['truematch'],df['prediction'],t)
         for t in thresholds],columns=['Precision','Recall','F-score'])
        for df in [df[df['mlpurpose']=='check'] for (_,df) in probit_results]]
    probit_performance = [df.assign(Threshold=thresholds,Model=m)
                          for (df,m) in zip(probit_performance,
                                            ['Full','Amafin','Amasub'])]
    probit_performance = pd.concat(probit_performance,axis=0)
    optimal_p = dict(probit_performance.set_index('Threshold').groupby(
        'Model')['F-score'].idxmax())
    pd.Series(optimal_p,name='p').to_pickle(
        pars.paths.data.proc+'firmnames/'+'series_F1_optimal_p.gz')
    probit_performance.to_pickle(pars.paths.data.proc+'firmnames/'
                                 +'dataframe_probit_performance.gz')
    ### Final decision
    df_results_full = probit_results[0][1]
    df_decision = df_results_full.assign(
        decision_model=df_results_full['prediction']>optimal_p['Full'])
    df_decision.loc[:,'decision'] = df_decision['decision_model']
    df_decision.loc[~df_decision['nonexact'],'decision'] = True
    df_decision.loc[df_decision['randomlabelled'],
                    'decision'] = df_decision['truematch']
    df_decision.to_pickle(pars.paths.data.proc+'firmnames/'
                          +'dataframe_match_decision.gz')

def fuzzy_links():
    """
    Build dataframe of links established with the fuzzy matching 
    algorithm. Includes identifying information of the firms and patent
    applicants.
    """
    ### Parameters
    dict_cols = {'SUBS_BVDEPNR':'id','SUBS_NAME':'nm','SUBS_CNTRY':'cntry',
                 'IDNR':'id','NAME':'nm','CNTRYCDE':'cntry'}
    ### Data
    df_decision = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                                 +'dataframe_match_decision.gz')[
                                     ['link_id','decision']]
    df_candidates = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'+'matching_results_all.gz')[
            ['link_id','m_train','name_train','name_match']]
    df_amafin = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'+'names_amafin5.gz').drop(
            columns='NAME_NAT').rename(columns=dict_cols)
    df_amasub = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                               +'names_amasub5.gz').rename(columns=dict_cols)
    df_amafin.loc[:,'src'] = 'NAME'
    df_amasub.loc[:,'src'] = 'SUBS_NAME'
    df_ama = pd.concat([df_amafin,df_amasub],axis=0)
    df_ps = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'+'names_ps_tls206_5.gz')[
            ['person_id', 'person_name']]
    # Build df with all found links
    df_match = df_candidates[df_candidates['link_id'].isin(
        df_decision[df_decision['decision']==True]['link_id'])]
    df_match_ama = pd.merge(
        left=df_ama,
        right=df_match,
        left_on=['nm','src'],
        right_on=['name_train','m_train'],
        how='inner')
    df_match_all = pd.merge(
        left=df_ps,
        right=df_match_ama,
        left_on='person_name',
        right_on='name_match',
        how='inner')
    df_match_all = df_match_all[
        ['person_id','person_name','id','nm','cntry','src','link_id']]
    # Save table
    df_match_all.to_pickle(pars.paths.data.proc+'firmnames/'
                           +'fuzzy_matches_all.gz')
    df_match_all.to_csv(pars.paths.data.proc+'firmnames/'
                        +'fuzzy_matches_all.csv')

def exact_links():
    """
    Find all exact matches.
    """
    ### Read in and select data
    print('Reading in data')
    print('\rPATSTAT [ ]','AMAFIN [ ]','AMASUB [ ]',sep='   ',end='')
    df_ps = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                           +'names_ps_tls206_5.gz')
    df_ps_names = df_ps[
        ['person_name','doc_std_name','psn_name','han_name']].reset_index(
            drop=True).reset_index().rename(columns={'index': 'row_id'})
    del df_ps
    df_ps_proc = pd.read_pickle(pars.paths.data.proc
                                +'firmnames/names_ps_tls206_proc.gz')
    df_ps_names_proc = df_ps_proc[
        ['person_name','doc_std_name','psn_name','han_name']].reset_index(
            drop=True).reset_index().rename(columns={'index': 'row_id'})
    del df_ps_proc
    print('\rPATSTAT [X]','AMAFIN [ ]','AMASUB [ ]',sep='   ',end='')
    df_amafin = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                               +'names_amafin5.gz')
    df_amafin_name = df_amafin[['IDNR','NAME']]
    del df_amafin
    df_amafin_proc = pd.read_pickle(pars.paths.data.proc
                                    +'firmnames/names_amafin_proc.gz')
    df_amafin_name_proc = df_amafin_proc[['IDNR','NAME']]
    del df_amafin_proc
    print('\rPATSTAT [X]','AMAFIN [X]','AMASUB [ ]',sep='   ',end='')
    df_amasub = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                               +'names_amasub5.gz')
    df_amasub_name = df_amasub[['SUBS_BVDEPNR','SUBS_NAME']]
    del df_amasub
    df_amasub_proc = pd.read_pickle(pars.paths.data.proc
                                    +'firmnames/names_amasub_proc.gz')
    df_amasub_name_proc = df_amasub_proc[['SUBS_BVDEPNR','SUBS_NAME']]
    del df_amasub_proc
    print('\rPATSTAT [X]','AMAFIN [X]','AMASUB [X]',sep='   ',end='\n')
    
    ### Exact merges
    print('Merging')
    print('\r[0/16]',end='')
    i = 1
    for proc,ama_name,df_ama in zip([False,False,True,True],
                                    ['NAME','SUBS_NAME']*2,
                                    [df_amafin_name,
                                     df_amasub_name,
                                     df_amafin_name_proc,
                                     df_amasub_name_proc]):
        if proc:
            df_ps = df_ps_names_proc
        else:
            df_ps = df_ps_names
        for ps_name in ['person_name','doc_std_name','psn_name',
                        'han_name']:
            buffer_df_merge = (pars.paths.data.proc+'firmnames/'+'merge_exact_'
                               +ama_name+'_'+ps_name+('_proc' if proc else '')
                               +'.gz')
            df_merge = pd.merge(
                left = df_ama,
                right = df_ps[['row_id',ps_name]],
                left_on = ama_name,
                right_on = ps_name,
                how = 'inner')
            # Save merged df
            df_merge.to_pickle(buffer_df_merge)
            print('\r['+str(i)+'/16]',end='')
            i += 1
    print('',end='\n')

def all_links():
    """
    Add exact links to the fuzzy links.
    
    Some BVD_EPNRs (subsidiary IDs) have two firm names attached. This
    causes duplicates in the IDs.
    """
    ### Parameters
    dict_cols = {'IDNR':'id','NAME':'nm','CNTRYCDE':'cntry',
                 'SUBS_BVDEPNR':'id','SUBS_NAME':'nm','SUBS_CNTRY':'cntry'}
    ### Data
    df_fuzzy = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                              +'fuzzy_matches_all.gz')
    df_exact_fin = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'
        +'merge_exact_NAME_person_name_proc.gz')[
            ['IDNR','person_name']].rename(columns=dict_cols)
    df_exact_sub = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'
        +'merge_exact_SUBS_NAME_person_name_proc.gz')[[
            'SUBS_BVDEPNR','SUBS_NAME','person_name']].rename(
                columns=dict_cols)
    df_amafin = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'+'names_amafin5.gz')[
            ['IDNR','NAME','CNTRYCDE']].rename(columns=dict_cols)
    df_amasub = pd.read_pickle(
        pars.paths.data.proc+'firmnames/'+'names_amasub5.gz')[
            ['SUBS_BVDEPNR','SUBS_NAME','SUBS_CNTRY']].rename(
                columns=dict_cols)
    df_ps_names = pd.read_pickle(
        pars.paths.data.proc+'firmnames/names_ps_tls206_proc.gz')[
            ['person_id','person_name']]
    # Manipulate
    df_exact_fin = df_exact_fin.merge(
        df_amafin,
        on='id',
        how='left').drop_duplicates(
            ['id','nm','person_name']).merge(
                df_ps_names,
                on='person_name',
                how='left').assign(
                    src='NAME',
                    link_id=range(5*10**6,len(df_exact_fin)+5*10**6))
    df_exact_sub = df_exact_sub.merge(
        df_amasub,
        on=['id','nm'],
        how='left').drop_duplicates(
            ['id','nm','person_name']).merge(
                df_ps_names,
                on='person_name',
                how='left').assign(
                    src='SUBS_NAME',
                    link_id=range(6*10**6,len(df_exact_sub)+6*10**6))
    df_exact = pd.concat([df_exact_fin,df_exact_sub],axis=0)
    ### Combine match links (fuzzy and exact)
    df_add = pd.merge(
        df_exact,
        df_fuzzy.assign(method='Fuzzy'),
        on=['id','person_id','src'],
        how='left',
        suffixes=('','_right'))
    df_add = df_add[df_add['method'].isna()][df_fuzzy.columns]
    df_all = pd.concat([df_fuzzy,df_add],axis=0)
    df_all.to_pickle(pars.paths.data.proc+'firmnames/'
                     +'matches_all_fuzzy_exact.gz')
    df_all.to_csv(pars.paths.data.proc+'firmnames/'
                  +'matches_all_fuzzy_exact.csv')

def groupfirms():
    """
    Define firm groups. Groups based on co-occurence in link to patent
    applicant.
    """
    ### Data
    df_match = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                              +'matches_all_fuzzy_exact.gz')
    ### Identify communities
    dfm = df_match[['id','person_id']].drop_duplicates()
    # Separate singletons
    bool_single = pd.merge(
        dfm,
        (dfm.groupby('person_id')['id'].apply(len)==1).rename('b'),
        left_on='person_id',
        right_index=True,
        how='left')['b']
    df_multiple = dfm[~bool_single]
    singletons = dfm[bool_single]['id'].drop_duplicates()
    singletons = singletons[~singletons.isin(df_multiple['id'])]
    # Construct edges
    def combs(x):
        out = pd.DataFrame(combinations(x,2),columns=['source','target'])
        return out
    df_edge = df_multiple.groupby('person_id')['id'].apply(combs)    
    df_edge = df_edge.groupby(['source','target']).size().rename(
        'weight').reset_index()
    G = nx.from_pandas_edgelist(df_edge,edge_attr='weight')
    # Communities
    comms = nx.community.louvain_communities(G, weight='weight', seed=12)
    # Finds the same communities:
    # comms1 = nx.community.greedy_modularity_communities(G,weight='weight')
    df_comm = pd.concat(
        [pd.DataFrame({'community_id':i,'firm_id':list(comm)}) for i,comm in
         enumerate(comms)],
        axis=0)
    max_id = df_comm.community_id.max()
    df_comm_single = pd.DataFrame(
        {'community_id':range(max_id+1,max_id+1+len(singletons)),
         'firm_id':singletons})
    df_comm_all = pd.concat([df_comm,df_comm_single],axis=0)
    ### Disambiguation
    # Some patent applicants are linked to multiple communities
    # Keep links to the largest community
    df_match_comms = pd.merge(
        df_match,
        df_comm_all.rename(columns={'firm_id':'id'}),
        on='id',
        how='left')
    dups = df_match_comms.groupby('person_id')['community_id'].nunique()
    dups = dups[dups>1].index
    df_ok = df_match_comms[~df_match_comms.person_id.isin(dups)]
    df_dup = df_match_comms[df_match_comms.person_id.isin(dups)]
    df_dup_count = df_dup.groupby(['person_id','community_id'])[
        'community_id'].count().rename('comms_count').reset_index(
            ).sort_values(['person_id','comms_count'],ascending=False)
    df_dup_count.loc[:,'keepme'] = ~df_dup_count['person_id'].duplicated(
        keep='first')
    df_dup = pd.merge(
        df_dup,
        df_dup_count,
        on=['person_id','community_id'],
        how='left')
    df_dup = df_dup[df_dup['keepme']]
    df_comm_disamb = pd.concat([df_ok,df_dup[df_ok.columns]],axis=0)    
    # Save to disk
    df_comm_all.to_pickle(pars.paths.data.proc+'firmnames/'
                          +'firm_groups_patent_communities.gz')
    df_comm_disamb.to_pickle(pars.paths.data.proc+'firmnames/'
                             +'matches_communities.gz')

def patentconsolidatedlink():
    """
    Use the communities to link each patent applicant to the dominant
    consolidated firm in the community.
    """
    ### Parameters
    size_vars = ['EMPL','TOAS','TURN']
    ### Data
    df_match_comm = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                                   +'matches_communities.gz')
    df_amafin = imp.import_ama(
        table='fin',
        cols=['IDNR','REPBAS','ACCPRA','MONTHS','CURRENCY','CLOSDATE_year',
              'CLOSDATE','LSTATUS','STATUSDATE','LASTYEAR','QUOTED','CONSOL']
        +size_vars)
    ### Find dominant consolidated firm
    # Find link community<->dominant firm. Not existing for each community.
    df_amafin_select = df_amafin[
        (df_amafin['REPBAS']=='Consolidated data') &
        (df_amafin['MONTHS']==12) &
        (df_amafin['IDNR'].isin(df_match_comm['id'].unique()))].sort_values(
            ['ACCPRA','CLOSDATE'],ascending=False).drop_duplicates(
                subset=['IDNR','CLOSDATE_year','ACCPRA','CLOSDATE'])
    df_amafin_large = df_amafin_select.sort_values(
        by=size_vars,ascending=False).drop_duplicates(
            subset='IDNR')
    df_links = pd.merge(
        df_match_comm[['community_id','id']].drop_duplicates(),
        df_amafin_large,
        left_on='id',
        right_on='IDNR',
        how='inner').sort_values(
            ['community_id']+size_vars,ascending=False).drop_duplicates(
                subset='community_id')[
                    ['community_id','id']]
    ### Save to disk
    df_links.to_pickle(pars.paths.data.proc+'firmnames/'
                       +'links_community_dominant_cons_firm.gz')

def patentfirmlink():
    """
    Find the 1-on-1 link between patent applicants and firms. Match the
    patent applicant to the firm with the highest p-hat from the fuzzy
    matching exercise. Break ties.
    """
    # NOTE: There are duplicates within id-person_id-src as some
    # subsidiaries have two names ("... via its funds") per firm ID.
    ### Parameters
    epsilon = 0.01
    ### Data
    df_match = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                              +'matches_all_fuzzy_exact.gz')
    df_decision = pd.read_pickle(pars.paths.data.proc+'firmnames/'
                                 +'dataframe_match_decision.gz')
    df_amafin = imp.import_ama(
        table='fin',
        cols=['IDNR','REPBAS','ACCPRA','CLOSDATE_year','EMPL','TOAS','TURN'])
    df_amasub = imp.import_ama(
        table='sub',
        cols=['SUBS_BVDEPNR','SUBS_DATE','SUBS_EMPL','SUBS_TOAS','SUBS_OPRE'])
    ### Select unique links
    df = pd.merge(
        df_match,
        df_decision,
        on='link_id',
        how='left')
    # Find candidates (within epsilon of max p-hat; to allow for
    # numerical imprecision)
    df = pd.merge(
        df,
        df.groupby('person_id')['prediction'].max().rename('prediction_max'),
        on='person_id',
        how='left')
    df.loc[:,'candidate'] = df['prediction'] > (df['prediction_max'] - epsilon)
    df.loc[df['link_id']>4*10**6,'candidate'] = True# Fix for exact matches
    df.loc[:,'random_filter'] = np.random.rand(len(df))
    df = df[df['candidate']]
    # Link to Amadeus data for firm info
    df_fin = df[df['src']=='NAME']
    df_sub = df[df['src']=='SUBS_NAME']
    df_amafin_select = df_amafin[
        (df_amafin['REPBAS']!='Consolidated data')].sort_values(
            ['EMPL','TOAS','TURN','CLOSDATE_year'],
            ascending=False).drop_duplicates('IDNR')
    df_amasub_select = df_amasub.sort_values(
        ['SUBS_EMPL','SUBS_TOAS','SUBS_OPRE'],
        ascending=False).drop_duplicates('SUBS_BVDEPNR')
    df_fin = pd.merge(
        df_fin,
        df_amafin_select,
        left_on='id',
        right_on='IDNR',
        how='inner')
    df_fin = df_fin.sort_values(
        ['EMPL','TOAS','TURN','random_filter'],
        ascending=False).drop_duplicates('person_id')
    df_sub = pd.merge(
        df_sub,
        df_amasub_select,
        left_on='id',
        right_on='SUBS_BVDEPNR',
        how='inner')
    df_sub = df_sub.sort_values(
        ['SUBS_EMPL','SUBS_TOAS','SUBS_OPRE','random_filter'],
        ascending=False).drop_duplicates('person_id')
    # Pull and save
    df_all = pd.concat([df_fin,df_sub],axis=0)[df_match.columns]
    df_all.to_pickle(pars.paths.data.proc+'firmnames/'
                     +'links_1on1_firms_patent_applicants.gz')

def main():
    """
    Runs this module: performs firm name matching.
    
    Best to run step-by-step. Requires running the matching algorithm
    separately per Amadeus database. Matching best run in parts to
    control process.
    """
    ## Process names
    if False:
        df_ps = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                               +'names_ps_tls206_5.gz')
        df_ps_proc = proc(
            df=df_ps,cols=['person_name','doc_std_name','psn_name','han_name'])
        df_ps_proc.to_pickle(pars.paths.data.proc
                             +'firmnames/names_ps_tls206_proc.gz')
        df_amafin = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                                   +'names_amafin5.gz')
        df_amafin_proc = proc(df=df_amafin,cols=['NAME','NAME_NAT'])
        df_amafin_proc.to_pickle(pars.paths.data.proc
                                 +'firmnames/names_amafin_proc.gz')
        df_amasub = pd.read_pickle(pars.paths.data.proc + 'firmnames/'
                                   +'names_amasub5.gz')
        df_amasub_proc = proc(df=df_amasub,cols=['SUBS_NAME'])
        df_amasub_proc.to_pickle(pars.paths.data.proc
                             +'firmnames/names_amasub_proc.gz')
        ### Save mappings original <-> processed names
        # Ama Fin
        df_amafin_mapping_names = pd.merge(
            left=df_amafin[['NAME','NAME_NAT']],
            right=df_amafin_proc[['NAME','NAME_NAT']],
            left_index=True,
            right_index=True,
            suffixes=(None,'_proc'))
        df_amafin_mapping_names.to_pickle(
            pars.paths.data.proc
            +'firmnames/names_amafin_mapping_original_proc.gz')
        df_amafin_mapping_names.to_csv(
            pars.paths.data.proc
            +'firmnames/names_amafin_mapping_original_proc.csv')
        # Ama Sub
        df_amasub_mapping_names = pd.merge(
            left=df_amasub[['SUBS_NAME']],
            right=df_amasub_proc[['SUBS_NAME']],
            left_index=True,
            right_index=True,
            suffixes=(None,'_proc'))
        df_amasub_mapping_names.to_pickle(
            pars.paths.data.proc
            +'firmnames/names_amasub_mapping_original_proc.gz')
        df_amasub_mapping_names.to_csv(
            pars.paths.data.proc
            +'firmnames/names_amasub_mapping_original_proc.csv')
        # PATSTAT
        df_ps_mapping_names = pd.merge(
            left=df_ps[['person_name','doc_std_name','psn_name',
                        'han_name']],
            right=df_ps_proc[['person_name','doc_std_name','psn_name',
                              'han_name']],
            left_index=True,
            right_index=True,
            suffixes=(None,'_proc'))
        # File too large to pickle at once: save per variable
        for var in ['person_name','doc_std_name','psn_name','han_name']:
            cols = list(df_ps_mapping_names.columns)
            df_ps_mapping_names_var = df_ps_mapping_names[
                [col for col in cols if col.startswith(var)]]
            df_ps_mapping_names_var.to_pickle(
                pars.paths.data.proc+'firmnames/'+
                'names_ps_mapping_'+var+'_original_proc.gz')
        
    ## Read in files
    if False:
        df_ps_proc = pd.read_pickle(pars.paths.data.proc
                                    +'firmnames/names_ps_tls206_proc.gz')
        df_amafin_proc = pd.read_pickle(pars.paths.data.proc
                                        +'firmnames/names_amafin_proc.gz')
        df_amasub_proc = pd.read_pickle(pars.paths.data.proc
                                        +'firmnames/names_amasub_proc.gz')
    
    ## Cosine Similarity matching
    # Vectorize
    if False:
        # Vectorizer
        vectorizer = TfidfVectorizer(
            min_df=2,# Smooth IDF results in IDF=1 for n=N=1. Hence unnecessary
            norm='l2',# L1 or L2 norm
            #analyzer=cossim.ngrams,# Own ngram function.
            ngram_range=(3,3),
            analyzer='char',
            dtype=np.float32,# Does not allow np.float16
            use_idf=True,# False: idf(t)=1
            smooth_idf=True,# +1 in both numerator and denominator
            sublinear_tf=False,# Default is False. Changes TF to (1 + log(TF))
            # strip_accents='unicode',# Could opt None if I trust own cleaning
            lowercase=False,# Opt for True if not already done
            binary=False,# TF term to 0,1
            )
        # Fit vectorizer on training data
        ser_train = (df_amafin_proc['NAME']
                      .drop_duplicates()
                      .dropna()
                      .reset_index(drop=True))
        # ser_train = (df_amasub_proc['SUBS_NAME']
        #               .drop_duplicates()
        #               .dropna()
        #               .reset_index(drop=True))
        csr_train = vectorizer.fit_transform(ser_train)
        # Transform matching data, using fitted vectorizer
        ser_other = (df_ps_proc['person_name']
                      .drop_duplicates()
                      .dropna()
                      .reset_index(drop=True))
        csr_other = vectorizer.transform(ser_other)
        # Save vocabulary
        dict_vocab = vectorizer.vocabulary_
        df_vocab = pd.DataFrame({'3-gram':list(dict_vocab.keys()),
                                 'ind':list(dict_vocab.values())})
        df_vocab.to_pickle(pars.paths.data.proc
                           +'firmnames/'+'vocab_NAME.gz')
        # df_vocab.to_pickle(pars.paths.data.proc
        #                    +'firmnames/'+'vocab_SUBS_NAME.gz')
        # Save sparse matrices and Series
        scsp.save_npz(pars.paths.data.proc
                      +'firmnames/mat_NAME_NAME.npz',
                      csr_train)
        # scsp.save_npz(pars.paths.data.proc
        #               +'firmnames/mat_SUBS_NAME_SUBS_NAME.npz',
        #               csr_train)
        scsp.save_npz(pars.paths.data.proc
                      +'firmnames/mat_NAME_person_name.npz',
                      csr_other)
        # scsp.save_npz(pars.paths.data.proc
        #               +'firmnames/mat_SUBS_NAME_person_name.npz',
        #               csr_other)
        ser_train.to_pickle(pars.paths.data.proc
                            +'firmnames/names_amasub_SUBS_NAME_proc.gz')
        # ser_other.to_pickle(pars.paths.data.proc
        #                     +'firmnames/names_ps_tls206_person_name_proc.gz')
    
    # Read in matrices and series
    if True:
        csr_train = scsp.load_npz(pars.paths.data.proc
                                  +'firmnames/mat_NAME_NAME.npz')
        ser_train = pd.read_pickle(pars.paths.data.proc
                                   +'firmnames/names_amafin_NAME_proc.gz')
        csr_other = scsp.load_npz(pars.paths.data.proc
                                  +'firmnames/mat_NAME_person_name.npz')
        
        csr_train = scsp.load_npz(pars.paths.data.proc
                                  +'firmnames/mat_SUBS_NAME_SUBS_NAME.npz')
        csr_other = scsp.load_npz(pars.paths.data.proc
                                  +'firmnames/mat_SUBS_NAME_person_name.npz')
        ser_train = pd.read_pickle(pars.paths.data.proc
                                   +'firmnames/names_amasub_SUBS_NAME_proc.gz')
        
        ser_other = pd.read_pickle(
            pars.paths.data.proc
            +'firmnames/names_ps_tls206_person_name_proc.gz')
    
    # Cosine similarity scores (takes multiple days)
    theta0 = 0.4*(math.pi/2)# share of quarter circle
    ii,jj,cc = [],[],[]
    nzz = []
    chunk_size = 1_000
    block_size = 50_000
    row_start,row_end = 6_550_000,6_560_000
    for (start,chunk) in chunker(csr_other[row_start:row_end],chunk_size):
        (i,j,c,nz) = cossim(chunk,start+row_start,csr_train,theta0)
        # Append to lists
        ii.extend(i)
        jj.extend(j)
        cc.extend(c)
        nzz.extend([nz])
        print('.',end='')
        # Save block of matches
        if (start%block_size==block_size-chunk_size or
            row_start+start+chunk_size>=csr_other.shape[0]):
            df_matches = pd.concat(
                [ser_other[ii].reset_index(drop=True).rename('ps_person_name'),
                 ser_train[jj].reset_index(drop=True).rename(
                     'amasub_SUBS_NAME'),
                 pd.Series(cc,name='theta',dtype=np.float32)],
                axis=1)
            df_matches.to_pickle(
                pars.paths.data.proc
                +'firmnames/matches_cossim_SUBS_NAME_person_name_'
                +str(row_start+start+chunk_size)+'_B.gz')
            ii,jj,cc = [],[],[]
            print(start+chunk_size)
    
    ### Pull all cossim matching results
    if False:
        files = os.listdir(pars.paths.data.proc+'firmnames')
        files = [file for file in files if not file.endswith('_MISTAKE.gz') and 
                 file.startswith('matches_cossim')]
        # Sort in order to align link_id with earlier created link_id
        # (At some point added two files. To preserve link between IDs
        # need to put last added files at end.)
        files_fix = [f for f in files if not f.endswith('9000.gz')] + [
            f for f in files if f.endswith('9000.gz')]
        # files_amasub = [file for file in files if 'SUBS_NAME' in file]
        # files_amafin = [file for file in files if 'SUBS_NAME' not in file]
        dfs_matches = []
        for file in files_fix:
            df_match = pd.read_pickle(pars.paths.data.proc+'firmnames/'+file)
            cols = list(df_match.columns)
            train_sources = ['NAME', 'SUBS_NAME']
            train_source = [src for src in train_sources if
                            file.startswith('matches_cossim_'+src)][0]
            col_train = [col for col in cols if col.endswith(train_source)][0]
            endrow = int(''.join([char for char in file if char.isdigit()]))
            df_match.loc[:,'m_train'] = train_source
            df_match.loc[:,'m_match'] = 'person_name'
            df_match.loc[:,'m_endrow'] = endrow
            df_match.rename(columns={'ps_person_name': 'name_match_proc',
                                     col_train: 'name_train_proc'},
                            inplace=True)
            dfs_matches.append(df_match)
        df_matches = pd.concat(dfs_matches,axis=0).reset_index().rename(
            columns={'index': 'm_index'})
        df_matches.loc[:,'match_id'] = range(len(df_matches))
        # Save these links (unique: name_match_proc,name_train_proc,m_train)
        df_matches.to_pickle(pars.paths.data.proc+'firmnames/'
                             +'matching_results_unique.gz')
        # Merge to original names
        df_links_ps = pd.read_pickle(pars.paths.data.proc+'firmnames/'+
        'names_ps_mapping_'+'person_name'+'_original_proc.gz').rename(
            columns={'person_name_proc': 'name_match_proc',
                     'person_name': 'name_match'}).drop_duplicates()
        df_links_amafin = pd.read_pickle(
            pars.paths.data.proc+'firmnames/'+
            'names_amafin_mapping_original_proc.gz')[
                ['NAME','NAME_proc']].rename(
                    columns={'NAME_proc': 'name_train_proc',
                             'NAME': 'name_train'})
        df_links_amafin.loc[:,'m_train'] = 'NAME'
        df_links_amasub = pd.read_pickle(
            pars.paths.data.proc+'firmnames/'+
            'names_amasub_mapping_original_proc.gz').rename(
                columns={'SUBS_NAME_proc': 'name_train_proc',
                         'SUBS_NAME': 'name_train'})
        df_links_amasub.loc[:,'m_train'] = 'SUBS_NAME'
        df_links_ama = pd.concat([df_links_amafin,df_links_amasub],
                                 axis=0).drop_duplicates()
        df_matches_original = df_matches.merge(
            right=df_links_ps,
            on='name_match_proc',
            how='left').merge(
                right=df_links_ama,
                on=['name_train_proc','m_train'],
                how='left')
        df_matches_original.loc[:,'same_start'] = (
            df_matches_original['name_match_proc'].str.slice(stop=1)==
            df_matches_original['name_train_proc'].str.slice(stop=1))
        df_matches_original.loc[:,'link_id'] = range(len(df_matches_original))
        df_matches_original.to_pickle(
            pars.paths.data.proc+'firmnames/'+'matching_results_all.gz')
        # Export to Excel for manual labelling
        df_matches_export = df_matches_original[
            ['link_id','same_start','theta','name_train','name_match']]
        start = 0
        while start<len(df_matches_export):
            step=100_000
            filename = (pars.paths.data.proc+'firmnames/'+'matching_results_'
                        +str(start+step)+'_A.xlsx')
            if os.path.exists(filename):
                raise ValueError('File already exists.')
            df_export = df_matches_export[start:start+step]
            df_export.to_excel(filename,index=False)
            start+=step
        # FIX: Export some later added matches separately (afterwards
        # manually added to last above file)
        df_matches_export[1_145_132:].to_excel(
            pars.paths.data.proc+'firmnames/'+'matching_results_'
            +str(1_200_000)+'_added.xlsx',index=False)
    
    ### Perform machine learning part
    if False:
        match_decision(update=False)
    
    ### Create dataframe with fuzzy-matched match info
    if False:
        fuzzy_links()
    ### Create dataframe with exact-matched match info
    if False:
        exact_links()
    ### Fuzzy and exact links
    if False:
        all_links()
    
    ### Create firm groups (patent applicant-to-firm group)
    if False:
        groupfirms()
    ### Link to community's dominant consolidated firm
    if False:
        patentconsolidatedlink()
    ### Create 1-on-1 link (patent applicant-to-1 firm)
    if False:
        patentfirmlink()

if __name__ == '__main__':
    main()
