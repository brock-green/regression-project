# Custom Modules
import os
import env
# Standard ds imports:
import pandas as pd
import numpy as np



def get_connection_url(db, user=env.user, host=env.host, password=env.password):
    """
    This function 1 positional arguement and kwargs for username, host, and password credentials from imported env module. Returns a formatted connection url to access mySQL database.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function uses the url from the get_connection_url() and reads the telco data from the Codeup db into a dataframe.
    '''
    sql_query = """
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, poolcnt, fips
            FROM properties_2017
            INNER JOIN predictions_2017 ON properties_2017.id = predictions_2017.id
            WHERE propertylandusetypeid = 261 
                AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection_url('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function checks if the 'zillow.csv' file exists in a local file path. It it exists it will read the file into a pandas df. If the file does not exist, it will used the new_telco_data() to read the telco data from Codeup db into a df and cache the file as 'telco.csv' in the local repository. Returns df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df