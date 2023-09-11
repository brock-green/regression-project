# Imports
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    # drop duplicates
    df = df.drop_duplicates()

    # rename columns
    rename_columns ={
        'bathroomcnt':'baths',
        'bedroomcnt':'beds',
        'calculatedfinishedsquarefeet':'sqft',
        'fips':'county',
        'taxvaluedollarcnt':'tax_value'}

    df = df.rename(columns=rename_columns)

    # handle nulls
    df = df.fillna(0)
    columns_to_replace = ['poolcnt']
    df[columns_to_replace] = df[columns_to_replace].applymap(lambda x: 'No Pool' if x == 0 else x)
    df = df[~(df == 0).any(axis=1)]
    df[columns_to_replace] = df[columns_to_replace].applymap(lambda x: '0' if x == 'No Pool' else x)
    
    # correct dtypes
    make_ints = ['beds','tax_value', 'yearbuilt', 'sqft', 'county', 'poolcnt']
    make_float = ['baths']
    
    for col in make_ints:
        df[col] = df[col].astype(int)
    for col in make_float:
        df[col] = df[col].astype(float)

    # handle outliers (top and bottom 0.5 %)

    # sort by tax_value
    z_sorted = df.sort_values(by='tax_value', ascending=True)

    # calculate the number of rows in the top and bottom 2%
    num_rows = int(0.005 * len(z_sorted))

    # remove the top and bottom .5% of rows
    z_outliers_removed = z_sorted.iloc[num_rows:-num_rows]

    # drop county
    df = z_outliers_removed.drop(columns=['county'])
    
    return df


############################## PreProcess ZILLOW FUNCTION ##############################

def preprocess(df):
    
    # split
    train, validate, test = split_function(df)
                                
    # scale
    train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test)
    
    # split into x and y
    x_train = train_scaled.drop(columns=['tax_value'])
    y_train = train_scaled.tax_value

    x_validate = validate_scaled.drop(columns=['tax_value'])
    y_validate = validate_scaled.tax_value

    x_test = test_scaled.drop(columns=['tax_value'])
    y_test = test_scaled.tax_value
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test, train_scaled, validate_scaled, test_scaled, train, validate, test


############################ SCALE ZILLOW FUNCTION ############################

# Scale
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['beds', 'baths','sqft', 'poolcnt'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

############################## PREPARE ZILLOW FUNCTION ##############################

    # df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})

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