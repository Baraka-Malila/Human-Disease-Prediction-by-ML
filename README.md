# Disease Prediction System

A machine learning-based web application that predicts potential diseases based on user-reported symptoms. The system uses a Support Vector Machine (SVM) model trained on a comprehensive dataset of symptoms and diseases.

## Features

- 🏥 Real-time disease prediction based on symptoms
- 🔍 Interactive symptom selection interface
- 🌓 Dark/Light mode support
- 📱 Responsive design for all devices
- ✅ Input validation and error handling
- ✨ Smooth animations and transitions

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Human-Disease-Prediction-by-ML.git
cd Human-Disease-Prediction-by-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface at: `http://localhost:5000`

## Documentation

Detailed documentation is available in the `docs` directory:

- [System Architecture](docs/architecture.md)
- [Installation Guide](docs/installation.md)
- [User Guide](docs/user-guide.md)
- [Technical Details](docs/technical.md)
- [API Documentation](docs/api.md)
- [Development Guide](docs/development.md)

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

## Features in Detail

### Symptom Selection
- Search functionality for symptoms
- Click-to-select interface
- Input validation
- Minimum three symptoms required

### Prediction System
- Real-time predictions
- Loading animations
- Error handling
- Medical disclaimers

### User Interface
- Modern, responsive design
- Dark/Light mode
- Smooth animations
- Mobile-friendly layout

## Contributing

Please read our [Development Guide](docs/development.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Machine Learning model trained using scikit-learn
- Web interface built with Flask
- Frontend styling with modern CSS
- Interactive features with vanilla JavaScript

## Support

For support, please:
1. Check the [documentation](docs/)
2. Submit an issue on GitHub
3. Contact the development team
