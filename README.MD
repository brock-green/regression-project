# Zillow:
<hr style="border:2px solid black">



## Goals: 
* Predict Property Tax assessed values ('taxcaluedollarcnt') of **Single Family Properties** that had a transaction during 2017. 
* Improve the performance of existing model



<hr style="border:2px solid black">

# Initial Thoughts
 
* Features related to size of the house 'sqft', 'beds', 'baths' are going to be predictive of tax value. They are going to be closely related so we may want to remove some to avoid multicolinearity.
 
<hr style="border:2px solid black"> 


# Data Dictionary
| Feature               | Definition |
|:----------------------|:-----------|
| beds| (float64) Number of bedrooms |
| baths | (float64) Number of bathrooms |
| sqft| (float64) finished square feet|
| yearbuilt| (int64) The year the house was built|
| poolcnt| (float64) If the property has a pool or not|



## Summary

- There are 56079 rows (homes)
- The target variable is `tax_value`, the assessed tax value of the home in dollars.
<hr style="border:2px solid black"> 


# The Plan
 
Plan --> Acquire --> Prepare --> Explore --> Model --> Deliver
 

#### Acquire
    * Use custom acquire module to create mySQL connection and read zillow dataset into pd.DataFrame
#### Prepare
    * Removed columns with duplicate information
    * Renamed Columns
    * Handled nulls
    * Changed dtypes
    * Split data into train, validate and test (approx. 60/20/20)
#### Explore
    * Vizualize data distributions of feature interaction with target
    * Perform stats testing on potential features
    * Choose features for the model
#### Model
    * OLS - Multiple Regression
        * Features: 'beds','baths','sqft', 'poolcnt', 'yearbuilt'
    * LassoLars
        * alpha=4
    * 2nd Degree Polynomial
    * GLM - Tweedie Regressor

<hr style="border:2px solid black"> 

# Steps to Reproduce
>1) Clone this repo.
>2) Create env.py file with credentials to access Codeup mySQL server
>3) Run notebook.
<hr style="border:2px solid black"> 
 

### <u>Next Steps:</u>
>* Zillow dataset includes many possible features. Do further feature engineering to better capture subsets of the data:
>    * Distressed properties: tax_delinquency, having more than 2x bedrooms to bathrooms...
>    * High-end properties: Can lot size help predict high-end properties when our features cannot?
>* Spend more time investigating outliers. Eliminating the top 0.5% and bottom 0.5% had the biggest positive impact on the model. Are there erros in the data or are our features missing part of the story? 
>    * Low-end outliers: Are these actually SFR homes? Are there actually structures or just empty lots?