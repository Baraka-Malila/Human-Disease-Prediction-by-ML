document.addEventListener('DOMContentLoaded', function() {
    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    const icon = themeToggle.querySelector('i');

    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    themeToggle.addEventListener('click', () => {
        const currentTheme = body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Symptom Search and Selection
    const symptomSearch = document.getElementById('symptomSearch');
    const symptomSuggestions = document.getElementById('symptomSuggestions');
    const selectedSymptoms = document.getElementById('selectedSymptoms');
    const predictButton = document.getElementById('predictButton');
    const loadingAnimation = document.getElementById('loadingAnimation');
    const predictionResult = document.getElementById('predictionResult');
    const resultContent = document.getElementById('resultContent');

    let validSymptoms = [];
    let selectedSymptomSet = new Set();

    // Fetch valid symptoms
    fetch('/valid-symptoms')
        .then(response => response.json())
        .then(data => {
            validSymptoms = data.symptoms;
        })
        .catch(error => console.error('Error fetching symptoms:', error));

    // Handle symptom search
    symptomSearch.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        if (searchTerm.length < 2) {
            symptomSuggestions.style.display = 'none';
            return;
        }

        const filteredSymptoms = validSymptoms.filter(symptom => 
            symptom.toLowerCase().includes(searchTerm) && 
            !selectedSymptomSet.has(symptom)
        );

        if (filteredSymptoms.length > 0) {
            symptomSuggestions.innerHTML = filteredSymptoms
                .map(symptom => `<div class="symptom-suggestion">${symptom}</div>`)
                .join('');
            symptomSuggestions.style.display = 'block';
        } else {
            symptomSuggestions.style.display = 'none';
        }
    });

    // Handle symptom selection
    symptomSuggestions.addEventListener('click', function(e) {
        if (e.target.classList.contains('symptom-suggestion')) {
            const symptom = e.target.textContent;
            addSymptom(symptom);
            symptomSearch.value = '';
            symptomSuggestions.style.display = 'none';
        }
    });

    function addSymptom(symptom) {
        if (!selectedSymptomSet.has(symptom)) {
            selectedSymptomSet.add(symptom);
            const tag = document.createElement('div');
            tag.className = 'symptom-tag';
            tag.innerHTML = `
                ${symptom}
                <span class="remove-symptom" data-symptom="${symptom}">&times;</span>
            `;
            selectedSymptoms.appendChild(tag);
            updatePredictButton();
        }
    }

    // Handle symptom removal
    selectedSymptoms.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove-symptom')) {
            const symptom = e.target.dataset.symptom;
            selectedSymptomSet.delete(symptom);
            e.target.parentElement.remove();
            updatePredictButton();
        }
    });

    function updatePredictButton() {
        predictButton.disabled = selectedSymptomSet.size < 3;
    }

    // Prediction History
    let predictionHistory = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    const historyList = document.getElementById('historyList');
    const historySearch = document.getElementById('historySearch');
    const historySort = document.getElementById('historySort');
    const newPredictionButton = document.getElementById('newPredictionButton');

    // Function to add prediction to history
    function addToHistory(prediction, symptoms) {
        const historyItem = {
            id: Date.now(),
            date: new Date().toISOString(),
            prediction: prediction,
            symptoms: symptoms
        };
        predictionHistory.unshift(historyItem);
        localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
        updateHistoryDisplay();
    }

    // Function to update history display
    function updateHistoryDisplay() {
        const searchTerm = historySearch.value.toLowerCase();
        const sortOrder = historySort.value;
        
        let filteredHistory = predictionHistory.filter(item => 
            item.prediction.toLowerCase().includes(searchTerm) ||
            item.symptoms.some(symptom => symptom.toLowerCase().includes(searchTerm))
        );

        if (sortOrder === 'oldest') {
            filteredHistory = filteredHistory.reverse();
        }

        historyList.innerHTML = filteredHistory.map(item => `
            <div class="history-item">
                <div class="history-item-header">
                    <span class="history-item-disease">${item.prediction}</span>
                    <span class="history-item-date">${new Date(item.date).toLocaleString()}</span>
                </div>
                <div class="history-item-symptoms">
                    ${item.symptoms.map(symptom => 
                        `<span class="history-item-symptom">${symptom}</span>`
                    ).join('')}
                </div>
            </div>
        `).join('');
    }

    // Event listeners for history filters
    historySearch.addEventListener('input', updateHistoryDisplay);
    historySort.addEventListener('change', updateHistoryDisplay);

    // New Prediction Button
    newPredictionButton.addEventListener('click', function() {
        // Clear current prediction
        selectedSymptomSet.clear();
        selectedSymptoms.innerHTML = '';
        symptomSearch.value = '';
        predictionResult.style.display = 'none';
        resultContent.innerHTML = '';
        
        // Hide new prediction button and show predict button
        newPredictionButton.style.display = 'none';
        predictButton.style.display = 'block';
        
        // Scroll to prediction section
        document.getElementById('prediction-section').scrollIntoView({ behavior: 'smooth' });
    });

    // Update prediction handler
    predictButton.addEventListener('click', async function() {
        if (selectedSymptomSet.size < 3) return;

        loadingAnimation.style.display = 'block';
        predictionResult.style.display = 'none';

        try {
            const symptomsString = Array.from(selectedSymptomSet).join(',');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    symptoms: symptomsString
                })
            });

            const data = await response.json();

            await new Promise(resolve => setTimeout(resolve, 2000));

            if (data.error) {
                resultContent.innerHTML = `<div class="error-message">${data.error}</div>`;
            } else {
                const prediction = typeof data.prediction === 'string' ? data.prediction : data.prediction[0];
                const matchedSymptoms = Array.isArray(data.matched_symptoms) ? data.matched_symptoms : [data.matched_symptoms];
                
                resultContent.innerHTML = `
                    <div class="prediction-success">
                        <h4>Predicted Disease:</h4>
                        <p>${prediction}</p>
                        <h4>Matched Symptoms:</h4>
                        <ul>
                            ${matchedSymptoms.map(symptom => `<li>${symptom}</li>`).join('')}
                        </ul>
                    </div>
                `;

                // Add to history
                addToHistory(prediction, matchedSymptoms);

                // Show new prediction button and hide predict button
                newPredictionButton.style.display = 'block';
                predictButton.style.display = 'none';
            }
        } catch (error) {
            console.error('Prediction error:', error);
            resultContent.innerHTML = '<div class="error-message">An error occurred. Please try again.</div>';
        } finally {
            loadingAnimation.style.display = 'none';
            predictionResult.style.display = 'block';
        }
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', function(e) {
        if (!symptomSearch.contains(e.target) && !symptomSuggestions.contains(e.target)) {
            symptomSuggestions.style.display = 'none';
        }
    });

    // Add medical disclaimer
    const disclaimer = document.createElement('div');
    disclaimer.className = 'medical-disclaimer';
    disclaimer.innerHTML = `
        <p><strong>Medical Disclaimer:</strong> This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
    `;
    document.querySelector('.prediction-container').appendChild(disclaimer);

    // Initial history display
    updateHistoryDisplay();
}); 