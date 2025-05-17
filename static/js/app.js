// Parking Detection App - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload functionality
    initFileUpload();
    
    // Initialize loading indicators
    initLoadingIndicators();
    
    // Initialize tooltips
    initTooltips();
});

function initFileUpload() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const fileNameDisplay = document.getElementById('file-name');
    
    // Handle click on upload area
    if (uploadArea) {
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
    }
    
    // Handle drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('active');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('active');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('active');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileName(e.dataTransfer.files[0].name);
            }
        });
    }
    
    // Handle file selection
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                updateFileName(fileInput.files[0].name);
            }
        });
    }
    
    // Update file name display
    function updateFileName(name) {
        if (fileNameDisplay) {
            fileNameDisplay.textContent = name;
            fileNameDisplay.style.display = 'block';
            
            // Show upload button
            const uploadButton = document.getElementById('upload-button');
            if (uploadButton) {
                uploadButton.style.display = 'block';
            }
        }
    }
    
    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Show loading indicator
            showLoading();
        });
    }
}

function initLoadingIndicators() {
    // Set up loading indicator functionality
    window.showLoading = function() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = 'block';
        }
    };
    
    window.hideLoading = function() {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }
    };
}

function initTooltips() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Function to handle API interaction
function analyzeImageViaAPI(formData) {
    showLoading();
    
    fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        hideLoading();
        
        if (data.success) {
            // Redirect to results page or update UI
            window.location.href = '/results';
        } else {
            showError(data.error || 'An error occurred during analysis');
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error: ' + error.message);
    });
}

function showError(message) {
    const alertArea = document.getElementById('alert-area');
    if (alertArea) {
        alertArea.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
    }
}
