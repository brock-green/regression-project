# Imports
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env
import os




############################## PREPARE ZILLOW FUNCTION ##############################

def prep_zillow(df):
    ''' 
    This function takes in a dataframe, renames the columns, and drops nulls. Also, it changes datatypes for appropriate columns and renames fips to the actual county names. Returns cleaned df.
    '''
    # drop duplicate columns and rows
    df = df.drop_duplicates()
    df = df.T.drop_duplicates().T
    df.rename(columns={'bathroomcnt' : 'baths',
                       'bedroomcnt' : 'beds',
                       'calculatedfinishedsquarefeet' : 'sqft',
                       'taxvaluedollarcnt' : 'taxvalue',
                       'fips' : 'county'}, inplace=True)
    df.replace(0, np.nan, inplace=True)
    df = df.dropna()
    make_ints = ['beds','sqft','taxvalue','yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    # df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    return df


############################ SPLIT ZILLOW FUNCTION ############################

def split_function(df):
    ''' 
    Function takes in 2 positional arguments for a dataframe and target variable. Returns train, validate, test, dataframes stratified on the target variable. Roughly (60/20/20) split.
    
    '''
    train, test = train_test_split(df,
                                   random_state=666,
                                   test_size=.20)
    
    train, validate = train_test_split(train,
                                   random_state=666,
                                   test_size=.25)
    return train, validate, test