import streamlit as st
import numpy as np
import pickle
import re

# Load the saved models
svc_model = pickle.load(open('svc_model.pkl', 'rb'))

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load feature columns
with open('features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Custom CSS to fix the title at the top, center it, and ensure responsiveness
st.markdown("""
    <style>
    .fixed-title {
        position: fixed;
        top: 5%;
        width: 100%;
        background-color: var(--primary-background-color);
        z-index: 9999;
        padding: 15px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex;
        justify-content: center;
        align-items: center;
        color: var(--primary-text-color);
        font-size: calc(1em + 1vw); /* Responsive font size */
        font-weight: bold;
        text-align: center;
    }
    .content {
        margin-top: 80px;
    }
    .bot-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App with fixed title
st.markdown('<div class="fixed-title"><h1>Disease Prediction Chatbot</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="content">', unsafe_allow_html=True)

# Initialize session state for input symptoms if not already present
if 'input_symptoms' not in st.session_state:
    st.session_state.input_symptoms = ''

# Initialize session state for initial greeting if not already present
if 'initial_greeting' not in st.session_state:
    st.session_state.initial_greeting = True

# SVG icon for the bot with escaped curly braces
bot_svg = """
<svg class="bot-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <circle cx="32" cy="32" r="32" fill="#f1f0f0"/>
    <path d="M32 4a28 28 0 1 0 28 28A28 28 0 0 0 32 4Zm0 52A24 24 0 1 1 56 32 24 24 0 0 1 32 56Z" fill="#d0d0d0"/>
    <circle cx="22" cy="24" r="4" fill="#000"/>
    <circle cx="42" cy="24" r="4" fill="#000"/>
    <path d="M22 36h20a10 10 0 0 1-20 0Z" fill="#000"/>
</svg>
"""

# Function to display bot message with avatar
def display_bot_message(message):
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start; margin-bottom: 10px;">
            {bot_svg}
            <div style="max-width: 60%; text-align: left; padding: 10px; background-color: #f1f0f0; border-radius: 10px; border: 1px solid #d0d0d0; color: #000;">
                <strong>Bot:</strong> {message}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display initial greeting
if st.session_state.initial_greeting:
    display_bot_message("Hello! I'm here to help you diagnose diseases. Please enter your symptoms, separated by commas, and I'll try to predict your condition. Feel free to enter at least three symptoms for more accurate results.")
    st.session_state.initial_greeting = False

# Function to process and predict disease based on symptoms
def process_and_predict():
    input_symptoms = st.session_state.input_symptoms
    if len(input_symptoms.strip()) == 0:
        st.warning('Please enter some symptoms.')
    else:
        symptoms_list = [symptom.strip().lower() for symptom in re.split(',|;', input_symptoms) if symptom.strip()]

        if len(set(symptoms_list)) < len(symptoms_list):
            st.warning('Please enter unique symptoms without any repetitions.')
        elif len(symptoms_list) < 3:
            st.warning('Please enter at least three symptoms for a more accurate prediction.')
        else:
            symptoms_dict = {symptom: 0 for symptom in feature_columns}

            for symptom in symptoms_list:
                if symptom in symptoms_dict:
                    symptoms_dict[symptom] = 1

            input_features = np.array(list(symptoms_dict.values())).reshape(1, -1)

            # Predict the disease
            prediction = svc_model.predict(input_features)
            prediction_label = le.inverse_transform(prediction)[0]

            # Display the prediction
            display_bot_message(f'Predicted Disease: {prediction_label}')

            # Clear the input field
            st.session_state.input_symptoms = ''

            # Prompt for next symptoms
            display_bot_message("Please enter your next symptoms separated by commas.")

# Input symptoms with Enter key submission
input_symptoms = st.text_input(
    'Enter your symptoms separated by commas (e.g., headache, fever, cough)',
    key='input_symptoms',
    on_change=process_and_predict
)

# Button for Predict (optional, for users who prefer clicking a button)
if st.button('Predict'):
    process_and_predict()

st.markdown('</div>', unsafe_allow_html=True)
