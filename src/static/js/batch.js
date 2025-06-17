document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('batchDropZone');
    const fileInput = document.getElementById('batchFileInput');
    const batchResults = document.querySelector('.batch-results');
    const progressBar = document.querySelector('.progress');
    const progressText = document.querySelector('.progress-text');
    const resultsGrid = document.querySelector('.results-grid');
    
    dropZone.addEventListener('click', () => fileInput.click());
    
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
        handleFiles(Array.from(e.dataTransfer.files));
    });
    
    fileInput.addEventListener('change', (e) => {
        handleFiles(Array.from(e.target.files));
    });

    async function handleFiles(files) {
        if (files.length === 0) return;
        
        batchResults.style.display = 'block';
        resultsGrid.innerHTML = '';
        let processed = 0;
        
        for (const file of files) {
            if (!file.type.match('image.*')) continue;
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                
                displayResult(file, data);
            } catch (error) {
                console.error(`Error processing ${file.name}:`, error);
            }
            
            processed++;
            updateProgress(processed, files.length);
        }
    }

    function updateProgress(current, total) {
        const percentage = (current / total) * 100;
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `Processing ${current}/${total} images`;
    }

    function displayResult(file, results) {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        
        const details = document.createElement('div');
        details.innerHTML = `
            <p><strong>File:</strong> ${file.name}</p>
            <p><strong>Diagnosis:</strong> ${results.predicted_class}</p>
            <p><strong>Confidence:</strong> ${(results.confidence * 100).toFixed(1)}%</p>
        `;
        
        card.appendChild(img);
        card.appendChild(details);
        resultsGrid.appendChild(card);
    }
});
