# Zillow:
<hr style="border:2px solid black">

## <u>Project Description</u>
words words words

## Goals: 
* goals 
* goals 
* goals


<hr style="border:2px solid black">

# Initial Thoughts
 
* Is there a price threshold where customer churn rate significantly increases?
* Can we identify actionable features that significantly increase churn rate?
 
<hr style="border:2px solid black"> 


# Data Dictionary
| Feature               | Definition |
|:----------------------|:-----------|
| blah| (int64) ID of the payment type. |
| iblah | (int64) ID of the internet service type. |
| blah| description|
| blah| description|
| blah| description|
| blah| description|
| blah| description a partner or not. |
| blah| description dependents or not. |
| blah| descriptionomer has been with the service. |


## Summary

- There are 7043 rows (customers)
- The target variable is `churn`, indicating whether a customer churned or not.
<hr style="border:2px solid black"> 


# The Plan
 
Plan --> Acquire --> Prepare --> Explore --> Model --> Deliver
 

#### Acquire
    * Use custom acquire module to create mySQL connection and read Telco_churn into pd.DataFrame
#### Prepare
    * Removed columns with duplicate information
    * Checked for nulls in the data (internet_service_type) and imputed for "No Internet Service"
    * Changed dtype for 'total_charges' to float
    * Reorganized 'payment_type' into values='Auto' and 'Manual'
    * Encoded categorical variables
    * Split data into train, validate and test (approx. 60/20/20), stratifying on 'churn'
#### Explore
    * Vizualize data distributions to identify potential drivers of churn
    * Perform stats testing on potential drivers of churn
    * Choose features for the model
#### Model
    * Decistion Tree Classifier - Iterate multiple models to tune hyperparameters
    * KNN - Iterate multiple models to tune hyperparameters
    * Logistic Regression

<hr style="border:2px solid black"> 

# Steps to Reproduce
>1) Clone this repo.
>2) Create env.py file with credentials to access Codeup mySQL server
>3) Run notebook.
<hr style="border:2px solid black"> 
 
### <u>Recommendations:</u>

>* Include tech support for all Fiber optic customers.
>* Incentivise customers to enroll in automatic payments. (i.e. rebate for enrollment)
>* Incentivise Month-to-Month customers to sign one or two year contracts.

### <u>Next Steps:</u>
>* Prescriptive Model: Identify the features that would be easiest for the business to take action. Then create a model using those features to find where executives can focus to lower churn rate.
>* Predictive Model: further EDA to find significant relationships for all potential features. Then create a model using all of the significant features.