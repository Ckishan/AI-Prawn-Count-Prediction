from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model, label_encoder = pickle.load(open("prawn_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        no_of_prawns = float(data['No_of_Prawns'])
        age = float(data['Age_of_Pond'])
        food = float(data['Food_Intake'])
        season = data['Season']

        # Encode season
        season_encoded = label_encoder.transform([season])[0]

        # Create feature array
        features = pd.DataFrame([[no_of_prawns, age, food, season_encoded]],
                                columns=['No_of_Prawns', 'Age_of_Pond', 'Food_Intake', 'Season'])

        # Predict
        prediction = model.predict(features)[0]
        return jsonify({'prediction': f"{prediction:.2f}"})  # Format to 2 decimal places

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
