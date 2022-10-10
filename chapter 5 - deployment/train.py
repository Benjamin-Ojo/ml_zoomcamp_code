# Importing packages
import numpy as np
import pandas as pd 

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importing pickle package.
import pickle

# Setting parameters.

# Logistic regression parameter.
c = 0.1

# Load dataset.
telco_data = pd.read_csv('data/telco_customer_churn.csv')

# Cleaning up columns names and categorical variables. 
columns = telco_data.columns.str.lower().str.replace(' ', '_')

# changeing columns names 
telco_data.columns = columns

# Categorical Features. 
categorical = ['customerid', 'gender', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod',
       'churn']

# Converting to lower case and removing space. 
for c in categorical: 
    telco_data[c] = telco_data[c].str.lower().str.replace(' ', '_')

# Converting totalcharge column to number. 
telco_data['totalcharges'] = pd.to_numeric(telco_data['totalcharges'], errors='coerce')

# Filling null values with 0. 
telco_data['totalcharges'] = telco_data['totalcharges'].fillna(0)

# converting churn to int
telco_data['churn'] = (telco_data['churn']== 'yes').astype(int)

# Setting up data validation dataset. 
telco_train, telco_test = train_test_split(telco_data, test_size = 0.20, random_state = 1)

telco_train = telco_train.reset_index()
telco_test = telco_test.reset_index()

# Creating y features. 
y_train = telco_train['churn'].values
y_test = telco_test['churn'].values

# Deleting churn from dataset. 
del telco_train['churn']
del telco_test['churn']

# Defining a training function. 
def train(df_train, df_y, c = 1.0):
    model = LogisticRegression(C= c , max_iter = 1000)
    dv = DictVectorizer(sparse = False)
    
    dict_train = df_train.to_dict(orient = 'records')
    x_train = dv.fit_transform(dict_train)
    
    model.fit(x_train, df_y)
    
    return model, dv

# Defining prediction function. 
def predict(df_val, model, dv): 
    dict_val = df_val.to_dict(orient = 'records')
    x_val = dv.transform(dict_val)
    
    y_pred = model.predict_proba(x_val)[:, 1]
    
    return y_pred

# Training model.
model, dv = train(telco_train, y_train, c= c)

# Model prediction.
y_pred = predict(telco_test,model, dv)

# Evaluating prediction with auc. 
auc = roc_auc_score(y_test, y_pred)

print(auc)

# file tile
churn_model = 'churn-model.bin'

# writting model to .bin file. 
with open(churn_model, 'wb') as f_out: 
    pickle.dump((model, dv), f_out)

