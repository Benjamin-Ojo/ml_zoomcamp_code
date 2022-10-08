#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing packages
import numpy as np
import pandas as pd 

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


# Load dataset.
telco_data = pd.read_csv('data/telco_customer_churn.csv')

# Checking dataframe.
telco_data.head()


# In[3]:


# Cleaning up columns names and categorical variables. 
columns = telco_data.columns.str.lower().str.replace(' ', '_')

# changeing columns names 
telco_data.columns = columns

# Checking for changes made. 
telco_data.columns


# In[ ]:





# In[4]:


# Categorical Features. 
categorical = ['customerid', 'gender', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod',
       'churn']

# Converting to lower case and removing space. 
for c in categorical: 
    telco_data[c] = telco_data[c].str.lower().str.replace(' ', '_')
    print(c)


# In[5]:


# Checking dataframe.
telco_data.head().T


# In[6]:


# Converting totalcharge column to number. 
telco_data['totalcharges'] = pd.to_numeric(telco_data['totalcharges'], errors='coerce')

# Filling null values with 0. 
telco_data['totalcharges'] = telco_data['totalcharges'].fillna(0)

# checking for null. 
telco_data['totalcharges'].isna().sum()


# In[7]:


# converting churn to int
telco_data['churn'] = (telco_data['churn']== 'yes').astype(int)

# Checking churn features.
telco_data.churn.value_counts()


# In[8]:


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


# In[9]:


telco_test


# In[10]:


# Defining a training function. 
def train(df_train, df_y, c = 1.0):
    model = LogisticRegression(C= c , max_iter = 1000)
    dv = DictVectorizer(sparse = False)
    
    dict_train = df_train.to_dict(orient = 'records')
    x_train = dv.fit_transform(dict_train)
    
    model.fit(x_train, df_y)
    
    return model, dv


# In[11]:


# Defining prediction function. 
def predict(df_val, model, dv): 
    dict_val = df_val.to_dict(orient = 'records')
    x_val = dv.transform(dict_val)
    
    y_pred = model.predict_proba(x_val)[:, 1]
    
    return y_pred


# In[12]:


# Setting parameters.

# Logistic regression parameter.
c = 0.1


# In[13]:


# Training model.
model, dv = train(telco_train, y_train, c= c)

# Model prediction.
y_pred = predict(telco_test,model, dv)

# Evaluating prediction with auc. 
auc = roc_auc_score(y_test, y_pred)

print(auc)


# In[14]:


customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75
}


# In[15]:


def predict_customer(customer, model, dv):
    x = dv.transform(customer)
    y_pred = model.predict_proba(x)[:, 1]
    return y_pred[0].round(3)


# In[16]:


# Predicting a single customer
result = predict_customer(customer, model, dv)

# Result of prediction.
result


# In[17]:


# Importing pickle package.
import pickle


# In[18]:


# file tile
model = f'model_C={c}.bin'


# In[19]:


# writting model to .bin file. 
with open(model, 'wb') as f_out: 
    pickle.dump((model, dv), f_out)

