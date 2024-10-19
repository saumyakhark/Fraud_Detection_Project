document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const loadingIndicator = document.getElementById('loading');
    const resultContainer = document.getElementById('result');
    const predictionSpan = document.getElementById('prediction');

    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the default form submission
        loadingIndicator.style.display = 'block'; // Show loading indicator
        resultContainer.style.display = 'none'; // Hide the result container

        // Create a FormData object to easily get the form data
        const formData = new FormData(form);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            // Assuming the server responds with the result
            predictionSpan.textContent = data; // Set prediction result
            resultContainer.style.display = 'block'; // Show the result container
            loadingIndicator.style.display = 'none'; // Hide loading indicator
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none'; // Hide loading indicator on error
            alert('There was an error processing your request.');
        });
    });
});
