from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import re

app = Flask(__name__)

# Load the model and encoders
with open('svc_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Convert feature_columns to list to maintain order
valid_symptoms = list(feature_columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/valid-symptoms')
def get_valid_symptoms():
    return jsonify({'symptoms': valid_symptoms})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        
        # Split the comma-separated string into a list
        symptom_list = [s.strip() for s in symptoms.split(',')]
        
        # Validate symptoms
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'})
        
        if len(symptom_list) < 3:
            return jsonify({'error': 'Please select at least 3 symptoms'})
        
        # Validate each symptom
        invalid_symptoms = [s for s in symptom_list if s not in valid_symptoms]
        if invalid_symptoms:
            return jsonify({
                'error': f'Invalid symptoms detected: {", ".join(invalid_symptoms)}'
            })
        
        # Create feature vector using the ordered feature_columns
        features = np.zeros(len(feature_columns))
        for symptom in symptom_list:
            if symptom in feature_columns:
                idx = feature_columns.index(symptom)
                features[idx] = 1
        
        # Make prediction
        prediction = svc_model.predict([features])[0]
        predicted_disease = le.inverse_transform([prediction])[0]
        
        # Get matched symptoms
        matched_symptoms = [symptom for symptom in symptom_list if symptom in valid_symptoms]
        
        return jsonify({
            'prediction': predicted_disease,
            'matched_symptoms': matched_symptoms
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True) 