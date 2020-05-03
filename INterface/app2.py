from flask import Flask, jsonify
from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
import sys
import traceback
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)
#xgb_model = pickle.load(open('test_finalized_random_forest_iris_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods= ['POST'])

def predict():
     #json_ = request.get_json(force= True)

     features = [float(x) for x in request.form.values()]

     final_features= [np.array(features)]#.tolist()
     #print(json_)
     #data_df = pd.DataFrame(json_)
     result = xgb_model.predict(final_features)

     # send back to browser
     output = round(result[0])#result.tolist()  #use .tolist to jsonify the list
     #int(result[0]) #{'results': int(result[0])}  #return 1 prediction at a time

     # return data
     #return jsonify(results=output)
     return render_template('index.html', prediction_text = 'Species is {}'.format(output))


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = xgb_model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    #try:
        #port = int(sys.argv[1]) # This is for a command-line argument
    #except:
        #port = 12345 # If you don't provide any port then the port will be set to 12345
    xgb_model = joblib.load('test_finalized_random_forest_iris_model.pkl') # Load "model.pkl"
    #print ('Model loaded')
    model_columns = joblib.load('model_columns.pkl') # Load "model_columns.pkl"
    #print ('Model columns loaded')
    app.run(debug=True)
