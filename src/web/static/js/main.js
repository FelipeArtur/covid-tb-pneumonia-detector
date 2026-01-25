document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const resultsSection = document.querySelector('.results-section');
    
    // Handle file selection via click
    dropZone.addEventListener('click', () => fileInput.click());
    
    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
    
    let totalPredictions = 0;
    let confidenceSum = 0;
    const diagnosisCount = {};
    
    // Update statistics
    function updateStatistics(predictedClass, confidence) {
        totalPredictions++;
        confidenceSum += confidence;
    
        diagnosisCount[predictedClass] = (diagnosisCount[predictedClass] || 0) + 1;
    
        const frequentDiagnosis = Object.keys(diagnosisCount).reduce((a, b) =>
            diagnosisCount[a] > diagnosisCount[b] ? a : b
        );
    
        document.getElementById('totalPredictions').textContent = totalPredictions;
        document.getElementById('frequentDiagnosis').textContent = frequentDiagnosis;
        document.getElementById('averageConfidence').textContent = 
            `${(confidenceSum / totalPredictions * 100).toFixed(1)}%`;
    
        document.querySelector('.statistics-section').style.display = 'block';
    }
    
    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image file');
            return;
        }
        
        // Show loading state
        dropZone.style.opacity = '0.5';
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        spinner.style.display = 'block';
        dropZone.appendChild(spinner);
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Add Grad-CAM toggle state to the request
        const gradcamToggle = document.getElementById('gradcamToggle');
        formData.append('gradcam', gradcamToggle.checked);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResults(file, data);
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            dropZone.style.opacity = '1';
            spinner.remove();
        });
    }
    
    function displayResults(file, results) {
        // Display image preview
        const previewImage = document.getElementById('previewImage');
        previewImage.src = URL.createObjectURL(file);
        
        // Handle Grad-CAM visualization
        const gradcamPreview = document.querySelector('.gradcam-preview');
        const gradcamImage = document.getElementById('gradcamImage');
        
        if (results.gradcam_image) {
            gradcamImage.src = results.gradcam_image;
            gradcamPreview.style.display = 'block';
        } else {
            gradcamPreview.style.display = 'none';
        }
        
        // Display prediction bars
        const barsContainer = document.querySelector('.prediction-bars');
        barsContainer.innerHTML = '';
        
        Object.entries(results.predictions).forEach(([className, probability]) => {
            const barHtml = `
                <div class="bar-container">
                    <div class="bar-label">
                        <span>${className}</span>
                        <span>${(probability * 100).toFixed(1)}%</span>
                    </div>
                    <div class="bar" style="width: ${probability * 100}%"></div>
                </div>
            `;
            barsContainer.innerHTML += barHtml;
        });
        
        // Update summary
        document.querySelector('.diagnosis').textContent = 
            `Diagnosis: ${results.predicted_class}`;
        document.querySelector('.confidence').textContent = 
            `Confidence: ${(results.confidence * 100).toFixed(1)}%`;
        
        // Update statistics
        updateStatistics(results.predicted_class, results.confidence);
        
        // Show results section
        resultsSection.style.display = 'block';
    }
});
