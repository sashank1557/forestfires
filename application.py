import pickle
from unittest import result
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = float(request.form['Classes'])
        region = float(request.form['Region'])

        # Create a DataFrame with the input values
        data = pd.DataFrame({
            'Temperature': [temperature],
            'RH': [rh],
            'Ws': [ws],
            'Rain': [rain],
            'FFMC': [ffmc],
            'DMC': [dmc],
            'ISI': [isi],
            'Classes': [classes],
            'Region': [region]
        })

        # Scale the input data
        scaled_data = standard_scaler.transform(data)

        # Make the prediction
        result = ridge_model.predict(scaled_data)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
