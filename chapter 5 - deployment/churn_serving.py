import pickle 
import numpy

from flask import Flask, request, jsonify

def predict_single_customer(customer, model, dv):
    x = dv.transform([customer])
    y_pred = model.predict_proba(x)[:,1]
    return y_pred[0]

with open('churn_model.bin', 'rb') as f_in:
    model, dv = pickle.load(f_in)
    
app = Flask('churn')
    
@app.route('/predict', method = ['POST'])

def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)