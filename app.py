from flask import Flask, render_template, request
import numpy as np 
import pandas as pd 
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Register a route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve all form values
    form_values = request.form.to_dict()

    # Remove the 'name' field from the form values
    name = form_values.pop('name', None)

    # Convert remaining values to integers
    input_features = [(x) for x in form_values.values()]

    features_values = [np.array(input_features)]
    features_name = ["bmi", "smoking", "alcohol", "stroke", "physical_health", "mental_health",
                     "diff_walking", "sex", "age_category", "diabetic", "physical_activity", "gen_health",
                     "sleep_time", "asthma", "kidney_disease", "skin_cancer"]

    df = pd.DataFrame(features_values, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "Heart Disease"
    else:
        res_val = "No Heart Disease"
    
    return render_template('index.html', prediction_text=f'{name} has {res_val}')

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
