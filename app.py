from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import re

app = Flask(__name__)

# Load the saved models
svc_model = pickle.load(open('svc_model.pkl', 'rb'))

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load feature columns
with open('features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Create a set of valid symptoms for quick lookup
valid_symptoms = set(feature_columns)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/valid-symptoms')
def get_valid_symptoms():
    # Return valid symptoms in alphabetical order
    return jsonify({'symptoms': sorted(list(valid_symptoms))})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        
        # Process symptoms
        symptoms_list = [symptom.strip().lower() for symptom in re.split(',|;', symptoms) if symptom.strip()]
        
        if len(symptoms_list) < 3:
            return jsonify({'error': 'Please enter at least three symptoms for a more accurate prediction.'})
            
        if len(set(symptoms_list)) < len(symptoms_list):
            return jsonify({'error': 'Please enter unique symptoms without any repetitions.'})
        
        # Validate symptoms
        invalid_symptoms = [symptom for symptom in symptoms_list if symptom not in valid_symptoms]
        if invalid_symptoms:
            return jsonify({
                'error': 'The following symptoms are not recognized: ' + ', '.join(invalid_symptoms) + 
                        '. Please use only valid symptoms from the list.'
            })
            
        # Create feature vector
        symptoms_dict = {symptom: 0 for symptom in feature_columns}
        for symptom in symptoms_list:
            symptoms_dict[symptom] = 1
                
        input_features = np.array(list(symptoms_dict.values())).reshape(1, -1)
        
        # Make prediction
        prediction = svc_model.predict(input_features)
        prediction_label = le.inverse_transform(prediction)[0]
        
        return jsonify({
            'prediction': prediction_label,
            'matched_symptoms': symptoms_list
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 