# Importing Package.
import requests

# Local webserver.
url = 'http://localhost:1998/predict'

# Customer info.
customer_id = "8879-zkjof"

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85 ** 88,
    "totalcharges": 33.75
}

# Sending request. 
print(requests.post(url, json = customer).json())

