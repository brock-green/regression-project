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
    This function uses the url from the get_connection_url() and reads the zillow data from the Codeup db into a dataframe.
    '''
    sql_query = """
SELECT *
FROM properties_2017
LEFT JOIN predictions_2017 ON properties_2017.id = predictions_2017.id
LEFT JOIN architecturalstyletype ON properties_2017.architecturalstyletypeid = architecturalstyletype.architecturalstyletypeid
LEFT JOIN buildingclasstype ON properties_2017.buildingclasstypeid = buildingclasstype.buildingclasstypeid
LEFT JOIN heatingorsystemtype ON properties_2017.heatingorsystemtypeid = heatingorsystemtype.heatingorsystemtypeid
LEFT JOIN storytype ON properties_2017.storytypeid = storytype.storytypeid
LEFT JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
LEFT JOIN typeconstructiontype ON properties_2017.typeconstructiontypeid = typeconstructiontype.typeconstructiontypeid
LEFT JOIN airconditioningtype ON properties_2017.airconditioningtypeid = airconditioningtype.airconditioningtypeid
LEFT JOIN unique_properties ON properties_2017.parcelid = unique_properties.parcelid
WHERE properties_2017.propertylandusetypeid = 261
    AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection_url('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function checks if the 'sfr_2017.csv' file exists in a local file path. It it exists it will read the file into a pandas df. If the file does not exist, it will used the new_zillow_data() to read the zillow data from Codeup db into a df and cache the file as 'sfr_2017.csv' in the local repository. Returns df.
    '''
    if os.path.isfile('sfr_2017.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('sfr_2017.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('sfr_2017.csv')
        
    return df