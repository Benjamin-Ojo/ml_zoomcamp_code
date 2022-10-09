# Importing pickle package.
import pickle

model = 'model_C=0.1.bin'

with open(model, 'rb') as f_in: 
    model, dv = pickle.load(f_in)
    
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

def predict_customer(customer, model, dv):
    x = dv.transform(customer)
    y_pred = model.predict_proba(x)[:, 1]
    return y_pred[0].round(3)

# Predicting a single customer
result = predict_customer(customer, model, dv)

# Result of prediction.
print(f'Churn Probability = {result}: Churn Decision = {result >= 0.5}')