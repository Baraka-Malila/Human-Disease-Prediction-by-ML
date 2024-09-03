# Human Disease Prediction System

## Overview

The Human Disease Prediction System uses machine learning algorithms to predict the likelihood of various diseases based on patient data. This system aims to assist healthcare professionals in diagnosing diseases more effectively by providing predictive insights based on historical medical data.

## Features

- **Predictive Modeling**: Uses advanced machine learning models to predict the likelihood of diseases.
- **Data Visualization**: Provides visualizations of the data and model predictions for better understanding.
- **User Interface**: Features a web interface for interacting with the prediction system.

## Project Structure

- `Deploying_model.py`: Script for deploying the trained model for predictions.
- `features.pkl`: Serialized file containing feature data used in model training.
- `Final.ipynb`: Jupyter Notebook with the final implementation and analysis of the prediction model.
- `label_encoder.pkl`: Serialized file containing label encoders used in the model.
- `svc_model.pkl`: Serialized Support Vector Classification model for predictions.
- `Testing.csv`: CSV file with testing data for model evaluation.
- `Training.csv`: CSV file with training data used to train the model.
- `static/`: Directory containing static assets such as images.

## How It Works

1. **Data Preparation**: The system uses `Training.csv` and `Testing.csv` to train and evaluate the model.
2. **Model Training**: The model is trained using various algorithms and saved as `svc_model.pkl`.
3. **Prediction**: The `Deploying_model.py` script loads the trained model and performs predictions on new data.
4. **Results**: Predictions are presented via the web interface or output files.

## Getting Started

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Baraka-Malila/Human-Disease-Prediction-by-ML.git
    ```

2. **Install Dependencies**:
    Ensure you have the required Python packages installed. You may use `requirements.txt` or `environment.yml` if available.

3. **Run the Application**:
    ```bash
    python Deploying_model.py
    ```

4. **Access the Web Interface**:
    Open your web browser and navigate to `http://localhost:5000` to interact with the application.

## Contributing

If you have suggestions or improvements, please fork the repository and submit a pull request.

## Contact

For any questions or support, please reach out to [Baraka Malila](mailto:your-email@example.com).
