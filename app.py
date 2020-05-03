from flask import Flask, jsonify
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import sys
import traceback
import pandas as pd
import numpy as np
app = Flask(__name__)
#@app.route('/predict', methods=['POST'])
@app.route('/predict', methods= ['POST'])



def predict():
     json_ = request.get_json(force= True)
     #print(json_)
     data_df = pd.DataFrame(json_)
     result = xgb_model.predict(data_df)

     # send back to browser
     output = result.tolist()  #use .tolist to jsonify the list
     #int(result[0]) #{'results': int(result[0])}  #return 1 prediction at a time

     # return data
     return jsonify(results=output)




if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    xgb_model = joblib.load('test_finalized_random_forest_iris_model.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
