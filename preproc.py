"""
Project:    MPhil thesis
Code:       Preprocessing of string data (preproc.py)
@author:    Leon Bremer
@date:      December 23, 2021

Preprocessing procedures for firm name (string) cleaning.
"""

import pandas as pd

import parameters as pars


def to_lower(vIn):
    """
    Use Panda's str.lower method to return the vector in lowercase.

    Parameters
    ----------
    vIn : pandas Series
        Vector of strings that need to be normalized.

    Returns
    -------
    vOut: pandas Series
        Pandas series of strings with punctuation.

    """    
    vOut = vIn.str.lower()
    
    return vOut


def delete_punc(vIn, punc=True, quotes=True, spaces=True, misc=False):
    """
    Delete punctuation from the character strings.

    Parameters
    ----------
    vIn : pandas Series
        Vector of strings that need to be normalized.
    
    punc: boolean
        Whether punctutation needs to be removed.
    
    quotes: boolean
        Whether quotes need to be removed.
    
    spaces: boolean
        Whether spaces need to be removed.
    
    ampersand: boolean
        Whether the ampersand needs to be removed.

    Returns
    -------
    vOut: pandas Series
        Pandas series of strings with punctuation.

    """
    chars_punc = ['.', ',', ';', ':', '!', '?', '\-', '/', '+', '\—']
    chars_quotes = ['`', '\'', '"',]
    chars_spaces = [' ', '\x1a', '\xa0', '\t']
    chars_misc = ['&','#','%','(',')','*','\[','\]','<','>','\\\\','{','}',
                  '\@','\^','_','\|','~']
    
    chars = []
    if punc:
        chars += chars_punc
    if quotes:
        chars += chars_quotes
    if spaces:
        chars += chars_spaces
    if misc:
        chars += chars_misc
    
    search_chars = ''.join(['['] + chars + [']'])
    
    vOut = vIn.str.replace(search_chars, '',regex=True,case=False)
    
    return vOut


def keep_alphanum(vIn, numeric=True, upper=True, space=False, keep_other=''):
    """
    Delete any character that is not alphanumeric. Provides options to 
    also preserve uppercase and numeric. Mind that this also drops
    letters with accents. In the most basic case only preserves a-z.
    
    Parameters
    ----------
    vIn : pandas Series
        Vector with string values to be cleaned.
    numeric : boolean, optional
        Whether to also preserve numeric characters 0-9.
    upper : boolean, optional
        Whether to also preserve uppercase letters A-Z.
    space : boolean, optional
        Whether to also preserve the space character ' '.
    keep_other : string, optional
        Provide a string of regex-compatible characters to also keep.
    
    Returns
    -------
    vOut: pandas Series
        pd.Series with clean string values.
    
    """
    chars_lower = ['a-z',]
    chars_upper = ['A-Z',]
    chars_num = ['0-9',]
    chars_space = [' ']
    
    chars = []
    chars += chars_lower
    if upper:
        chars += chars_upper
    if numeric:
        chars += chars_num
    if space:
        chars += chars_space
    
    search_chars = ''.join(['[^'] + chars + [keep_other] + [']'])
    
    vOut = vIn.str.replace(search_chars, '', regex=True)
    
    return vOut


def replace_german(vIn):
    """
    Map German characters to their German standardized normalization.

    Parameters
    ----------
    vIn : pandas Series
        Vector of character strings to be normalized. Only lowercase
        characters are recognized.

    Returns
    -------
    vOut : pandas Series
        Vector of adjusted characted strings.

    """
    char_mapping = {'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss'}
    search_keys = ''.join(['['] + list(char_mapping.keys()) + [']'])
    
    def char_map(s):
        s_out = char_mapping[s.group()]
        return s_out
    
    vOut = vIn.str.replace(search_keys, char_map, regex=True)
    
    return vOut


def replace_accented(vIn):
    """
    Replace characters with accents for their counterparts without
    accents. Only lowercase characters are recognized.
    
    This is based on a manual mapping based on a visual comparison
    between special characters and the common letters a-z. This is not a
    mapping based on linguistic information. E.g. the 'ö' in German is 
    pronounced as a 'oe', but here the 'ö' is simply stripped of its
    visual accents and mapped to a 'o'.

    Parameters
    ----------
    vIn : pandas Series
        Vector of character strings to be normalized. Should be
        lowercase.

    Returns
    -------
    vOut : pandas Series
        Vector of adjusted characted strings.

    """
    char_mapping = {'ä': 'a', 'à': 'a', 'á': 'a', 'ă': 'a', 'ą': 'a', 'ã': 'a',
                    'å': 'a', 'ā': 'a', 'ä': 'a', 'â': 'a',
                    'ß': 'b',
                    'ć': 'c', 'č': 'c', 'ç': 'c',
                    'đ': 'd', 'ď': 'd',
                    'ê': 'e', 'ě': 'e', 'ë': 'e', 'ё': 'e', 'ė': 'e', 'è': 'e',
                    'ę': 'e', 'é': 'e',
                    'ƒ': 'f',
                    'ğ': 'g',
                    'ī': 'i', 'î': 'i', 'í': 'i', 'ï': 'i', 'ì': 'i',
                    'ļ': 'l', #'ł̇': 'l', char causes error
                    'ń': 'n', 'ñ': 'n',
                    'ö': 'o', 'õ': 'o', 'ò': 'o', 'ó': 'o', 'ô': 'o',
                    'ř': 'r',
                    'ś': 's', 'ş': 's', 'š': 's',
                    'ţ': 't',
                    'ü': 'u', 'ů': 'u', 'ù': 'u', 'ú': 'u', 'ü': 'u', 'û': 'u',
                    'ū': 'u', 'µ': 'u', 'ū': 'u',
                    '×': 'x',
                    'ý': 'y', 'ÿ': 'y',
                    'ż': 'z', 'ž': 'z',
                    }
    search_keys = ''.join(['['] + list(char_mapping.keys()) + [']'])
    
    def char_map(s):
        s_out = char_mapping[s.group()]
        return s_out
    
    vOut = vIn.str.replace(search_keys, char_map, regex=True)
    
    return vOut


def replace_legal(ser_names, remove_full=False, remove_abbr=False):
    """
    Replace legal terms from the character strings with their
    abbreviated counterparts. Or choose to remove the legal terms. Full
    legal names are removed before abbreviated terms.

    Parameters
    ----------
    vIn : pandas Series
        Vector of character strings containing legal names. Must be
        named as the name variable.
    remove_full : boolean, default is False
        Deletes full legal term.
    remove_abbr : boolean, default is False
        Deletes abbreviated legal term.

    Returns
    -------
    vOut : pandas Series
        vector of adjusted character strings.

    """
    # Ad-hoc fix (German "beschränkter" in legal list, with accent)
    ser_names = ser_names.str.replace('beschraenkter','beschränkter',
                                      regex=False,case=False)
    
    bool_replace = not (remove_full or remove_abbr)
    df_legal = pd.read_excel(pars.paths.data.proc
                             +'legalnames/'
                             +'legalnames_for_manual_mapping_edit.xlsx',
                             sheet_name='manual_adjustment')
    df_legal.loc[:,]
    # Only keep the rows that have legal names that actually occur in 
    # that naming convention
    df_legal = df_legal[df_legal['source_col']==ser_names.name]
    if df_legal.empty:
        raise Exception('Series name likely invalid. Name:', ser_names.name)
    # Sort by length of legal names. First replace long, then short.
    df_legal.loc[:,'legal_length'] = df_legal['legal_terms'].str.len()
    df_legal.sort_values('legal_length',ascending=False,inplace=True)
    df_legal = df_legal[['legal_terms', 'legal_name_abbr']]
    
    ser_out = ser_names
    for (long,short) in df_legal.itertuples(index=False,name=None):
        if remove_full:
            ser_out = ser_out.str.replace(long,'',case=False,regex=False)
        if remove_abbr:
            ser_out = ser_out.str.replace(r'\b%s\b' % short,'',regex=True,
                                          case=False)
        if bool_replace:
            ser_out = ser_out.str.replace(long,short,case=False,regex=False)        
    
    return ser_out


def replace_legal1(vIn, remove_full=False, remove_abbr=False):
    """
    Replace legal terms from the character strings with their
    abbreviated counterparts. Or choose to remove the legal terms. Full
    legal names are removed before abbreviated terms.

    Parameters
    ----------
    vIn : pandas Series
        Vector of character strings containing legal names.
    remove_full : boolean, default is False
        Deletes full legal term.
    remove_abbr : boolean, default is False
        Deletes abbreviated legal term.

    Returns
    -------
    vOut : pandas Series
        vector of adjusted character strings.

    """
    df_legal = pd.read_csv(pars.paths.data.proc +
                         '/legalnames/hand_made_legal_entities_v1.csv')
    # Sort the 'from' terms from longest to smallest, so I replace long
    # terms before I parse through shorter sub-terms.
    df_legal['from_len'] = df_legal['from'].str.len()
    df_legal = df_legal.sort_values('from_len', ascending=False)
    
    # Loop over the mapped terms and replace values in vIn
    vOut = vIn
    for line in range(df_legal.shape[0]):
        sFrom = df_legal['from'][line]
        sTo = df_legal['to'][line]
        vOut = vOut.str.replace(sFrom, sTo, case=False, regex=False)
    
    return vOut


def ignore_legal(vIn, full=True, abbrev=False):
    """
    Ignore legal terms. Allows ignoring either/both full legal names
    (e.g. Encorporated) or/and abbreviated legal terms (e.g. Corp).
    
    The function ignores full legal names in order from longest to 
    shortest in order to capture all terms.
    
    Abbreviations get ignored after any full legal names are ignored.
    Also these terms are ignored in order from longest to shortest
    abbreviation.
    
    Abbreviations only get ignored when they have a leading space
    character. Otherwise it is likely that important information in the
    string gets deleted. But therefore it is also important to maintain
    a proper order in the sequence of string cleaning.
    """
    vOut = vIn
    df_legal = pd.read_csv(pars.paths.data.proc +
                         '/legalnames/hand_made_legal_entities_v1.csv')
    
    if full:
        # Sort
        df_legal['from_len'] = df_legal['from'].str.len()
        df_legal = df_legal.sort_values('from_len', ascending=False)
        
        # Replace
        for s in df_legal['from']:
            vOut = vOut.str.replace(s, '', case=False, regex=False)
    
    if abbrev:
        # Sort
        df_legal['to_len'] = df_legal['to'].str.len()
        df_legal = df_legal.sort_values('to_len', ascending=False)
        
        # Replace
        for s in df_legal['to']:
            vOut = vOut.str.replace(' ' + s, ' ', case=False, regex=False)
    
    return vOut


# def delete_accented(vIn):
#     """
#     Delete accents. Also deletes some other special characters.
#     Therefore not every characters gets mapped as desired, but it rather
#     gets deleted sometimes.
    
#     Arguments:
#         vIn: Input vector (pandas.Series) of strings
    
#     Return:
#         vOut: Output vector of cleaned strings
#     """
#     vOut = vIn.str.normalize(form='NFKD')
#     vOut = vOut.str.encode('ascii', errors='ignore')
#     vOut = vOut.str.decode('utf-8')
    
#     return vOut


def proc_multicol(df, cols, func, *args, **kwargs):
    """
    Perform fnc on the cols of df.
    
    This function performs a provided function on all provided columns
    of a provided dataframe.
    
    Mind that the function (func) needs to take a pd.Series as first
    argument.
    
    Arguments:
        df: Input dataframe
        cols: Columns to be manipulated with func
        func: Function to apply to cols
        *args: Non-keyword arguments to func
        **kwargs: Keyword arguments to func
    
    Return:
        df_out
    """
    df_out = df.apply(lambda c: func(c, *args, **kwargs)
                      if c.name in cols else c)
    
    return df_out