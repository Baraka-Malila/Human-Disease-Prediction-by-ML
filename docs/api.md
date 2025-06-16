# API Documentation

## Overview
The Disease Prediction System provides a RESTful API for disease prediction based on symptoms. All endpoints return JSON responses.

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. Get Valid Symptoms
Returns a list of all valid symptoms that can be used for prediction.

**Endpoint:** `GET /valid-symptoms`

**Response:**
```json
{
    "symptoms": [
        "symptom1",
        "symptom2",
        ...
    ]
}
```

### 2. Predict Disease
Predicts a disease based on provided symptoms.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "symptoms": "symptom1, symptom2, symptom3"
}
```

**Response:**
```json
{
    "prediction": "disease_name",
    "matched_symptoms": ["symptom1", "symptom2", "symptom3"]
}
```

**Error Response:**
```json
{
    "error": "Error message"
}
```

## Error Codes

### Common Error Messages
1. **Invalid Symptoms**
   ```json
   {
       "error": "The following symptoms are not recognized: symptom1, symptom2. Please use only valid symptoms from the list."
   }
   ```

2. **Insufficient Symptoms**
   ```json
   {
       "error": "Please enter at least three symptoms for a more accurate prediction."
   }
   ```

3. **Duplicate Symptoms**
   ```json
   {
       "error": "Please enter unique symptoms without any repetitions."
   }
   ```

## Rate Limiting
- 100 requests per minute per IP
- Rate limit headers included in response

## Authentication
Currently, no authentication is required for API endpoints.

## Best Practices

### Making Requests
1. Always validate symptoms before sending
2. Use proper error handling
3. Implement rate limiting on client side
4. Cache valid symptoms list

### Error Handling
1. Check for error responses
2. Implement proper retry logic
3. Handle network errors
4. Validate responses

## Example Usage

### Python
```python
import requests

# Get valid symptoms
response = requests.get('http://localhost:5000/valid-symptoms')
symptoms = response.json()['symptoms']

# Make prediction
prediction_data = {
    'symptoms': 'fever, headache, cough'
}
response = requests.post('http://localhost:5000/predict', json=prediction_data)
result = response.json()
```

### JavaScript
```javascript
// Get valid symptoms
fetch('http://localhost:5000/valid-symptoms')
    .then(response => response.json())
    .then(data => console.log(data.symptoms));

// Make prediction
fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        symptoms: 'fever, headache, cough'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Versioning
Current API version: 1.0

## Support
For API support or to report issues:
1. Check the documentation
2. Submit an issue on GitHub
3. Contact the development team 