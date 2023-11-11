from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np

app= Flask(__name__)

model = pickle.load(open('model1.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html', **locals())

@app.route('/predict',methods = ['POST','GET'])
def predict():
    gender = int(request.form.get('gender'))
    age = int(request.form.get('age'))
    hypertension = int(request.form.get('hypertension'))
    heart_diseases = int(request.form.get('heart_diseases'))
    smoking_history = int(request.form.get('smoking_history'))
    bmi = int(request.form.get('bmi'))
    HbA1c_level = float(request.form.get('HbA1c_level'))
    blood_glucose_level = int(request.form.get('blood_glucose_level'))

    result = model.predict_proba(np.array([gender,age,hypertension,heart_diseases,smoking_history,bmi,HbA1c_level,blood_glucose_level]).reshape(1, 8))[0]
    result1 = np.round(result[0] * 100, 2)
    result2 = np.round(result[1] *100, 2)
        
    return render_template('index.html', **locals())

if __name__ ==  '__main__':
    app.run(debug = True)
