# importing python packages. 
from flask import Flask

# Creating Flask instance.
app = Flask('ping')

# Creating flask rourt with python decorator.
@app.route('/ping',methods = ['GET'])

# Defining ping function.
def ping():
    return 'PONG'

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0',  port = 9695)