import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('models/model_act5.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # feature = request.form.values('spending')
    user_input = [float(request.form[field]) for field in ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c', 'blood_glucose_level']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    print(prediction)
    result = 'diabetic' if prediction == 1 else 'none diabetic'

    return render_template('index.html', prediction_output=f'Patient is {result}')


if __name__ == "__main__":
    app.run(debug=True)
