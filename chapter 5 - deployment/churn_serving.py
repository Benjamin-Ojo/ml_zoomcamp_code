# Import pickle 
import pickle

# Import flask packages
from flask import Flask
from flask import request
from flask import jsonify

# Loading model and dictvectorizer.
with open('churn-model.bin', 'rb') as f_in:
    model, dv = pickle.load(f_in)

# Creating Flask app.
app = Flask('predict')

# Creating single prediction. 
def predict_single_customer(customer, model, dv):
    x = dv.transform(customer)
    y_pred = model.predict_proba(x)[:, 1]
    
    return y_pred[0]

# Creating app route using decorator.
@app.route('/predict', methods = ['POST'])

def predict():
    customer = request.get_json()
    
    y_pred = predict_single_customer(customer, model, dv)
    
    result = { 'Churn Probability' :  float(y_pred), 
                'Churn' : bool(y_pred >= 0.50)}
    
    return jsonify(result)

if __name__ == '__main__': 
    app.run(host = '0.0.0.0', port = 1998)