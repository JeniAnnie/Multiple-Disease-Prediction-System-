from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heart')
def heart():
    return render_template('heart.html')

@app.route('/parkinsons')
def parkinsons():
    return render_template('parkinsons.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    data = [float(x) for x in request.form.values()]
    prediction = diabetes_model.predict([data])[0]
    result = 'Positive for Diabetes' if prediction == 1 else 'Negative for Diabetes'
    return render_template('diabetes.html', result=result)

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    data = [float(x) for x in request.form.values()]
    prediction = heart_model.predict([data])[0]
    result = 'Positive for Heart Disease' if prediction == 1 else 'Negative for Heart Disease'
    return render_template('heart.html', result=result)

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    data = [float(x) for x in request.form.values()]
    prediction = parkinsons_model.predict([data])[0]
    result = 'Positive for Parkinson’s Disease' if prediction == 1 else 'Negative for Parkinson’s Disease'
    return render_template('parkinsons.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
