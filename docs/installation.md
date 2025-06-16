# Installation Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Human-Disease-Prediction-by-ML.git
cd Human-Disease-Prediction-by-ML
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n disease-pred python=3.8
conda activate disease-pred
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import flask; import numpy; import pandas; import sklearn"
```

## Running the Application

### Development Mode
```bash
python app.py
```
The application will be available at `http://localhost:5000`

### Production Mode
```bash
gunicorn app:app
```

## Troubleshooting

### Common Issues

1. **Dependency Installation Fails**
   - Solution: Try updating pip: `pip install --upgrade pip`
   - Then reinstall requirements: `pip install -r requirements.txt`

2. **Port Already in Use**
   - Solution: Change the port in app.py or kill the process using the port

3. **Model Files Not Found**
   - Solution: Ensure all .pkl files are in the correct directory

4. **Python Version Issues**
   - Solution: Use Python 3.8 or higher as specified in requirements

## System Requirements

### Minimum Requirements
- CPU: 1 GHz
- RAM: 2 GB
- Storage: 500 MB
- OS: Windows 10/11, macOS, or Linux

### Recommended Requirements
- CPU: 2 GHz or higher
- RAM: 4 GB or higher
- Storage: 1 GB
- OS: Latest version of Windows, macOS, or Linux 