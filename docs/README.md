# Disease Prediction System Documentation

## Overview
The Disease Prediction System is a machine learning-based web application that helps predict potential diseases based on user-reported symptoms. The system uses a Support Vector Machine (SVM) model trained on a comprehensive dataset of symptoms and diseases.

## Table of Contents
1. [System Architecture](architecture.md)
2. [Installation Guide](installation.md)
3. [User Guide](user-guide.md)
4. [Technical Details](technical.md)
5. [API Documentation](api.md)
6. [Development Guide](development.md)

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`
3. Access the web interface at: `http://localhost:5000`

## Features
- Real-time disease prediction based on symptoms
- Interactive symptom selection interface
- Dark/Light mode support
- Responsive design for all devices
- Input validation and error handling
- Smooth animations and transitions

## Project Structure
```
Human-Disease-Prediction-by-ML/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/               # Static files
│   ├── css/             # Stylesheets
│   └── js/              # JavaScript files
├── templates/            # HTML templates
├── docs/                # Documentation
└── models/              # ML model files
    ├── svc_model.pkl    # Trained SVM model
    ├── features.pkl     # Feature columns
    └── label_encoder.pkl # Label encoder
```

## Contributing
Please read our [Development Guide](development.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 