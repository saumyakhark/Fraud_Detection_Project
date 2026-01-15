Fraud Detection Project
Overview-


This project is designed to detect fraudulent transactions using machine learning. The model predicts whether a given transaction is legitimate or fraudulent based on multiple features of the transaction data. The project is built using Python, and the dataset is preprocessed to retain only the most relevant features.


Tech Stacks used:

    1) Python: The core programming language used for building the project.
    2) Flask: Used for creating the web interface for the application.
    3) Pandas: For data manipulation and preprocessing.
    4) Scikit-learn: For building and evaluating the machine learning model.
    5) Jupyter Notebook: For development and experimentation with the model.
    6) Git: Version control system used for managing project files.
    7) Virtual Server: IBM LinuxOne Community Cloud for managing large datasets and deploying the model.

Project Structure: 

Fraud_Detection_Project/
│
├── app.py                     # Main Flask app
├── prepro.py                  # Script for data preprocessing
├── model_development.py       # Model training and evaluation
├── templates/
│   ├── index.html             # HTML template for the main page
│   └── result.html            # HTML template for displaying results
├── static/
│   └── styles.css             # CSS file for styling the web pages
├── data/
│   ├── creditcard.csv         # dataset used (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
│   ├── preprocessed_train_features.csv  # Preprocessed training features
│   ├── preprocessed_train_target.csv    # Preprocessed training labels
│   └── preprocessed_test_features.csv   # Preprocessed test features
└── README.md                  # This file

Features

The application includes:

    Transaction Analysis: Users can submit transaction data for analysis.
    Fraud Prediction: The machine learning model predicts whether the transaction is fraudulent or legitimate.
    Responsive UI: The web interface is mobile-friendly and aesthetically pleasing.

Instructions to Use

    Clone the repository:

    bash:-

git clone https://github.com/yourusername/Fraud_Detection_Project.git

Install the required dependencies: Navigate to the project directory and install the necessary libraries using pip: 

bash:-

pip install -r requirements.txt

Set up the project:

    Ensure that the dataset files (creditcard.csv, etc.) are placed in the data/ directory.
    Run the prepro.py script to preprocess the dataset and generate the necessary training and testing files.

Train the Model:

    Run model_development.py to train the machine learning model.
    The trained model will be saved in the directory for later use.

Run the Web Application: Start the Flask application:

bash :-

    python app.py

    The app will be accessible at http://127.0.0.1:5000/.

    Usage:
        Visit the web app.
        Submit transaction details (Transaction Amount, Transaction Time, User ID, Merchant ID, etc.).
        The model will classify the transaction as either Legitimate or Fraudulent.

Data

    Dataset: The dataset used in this project is the Kaggle Credit Card Fraud Detection Dataset.
    Preprocessing: Only 10 features are retained for prediction, including Transaction Amount, Transaction Time, User ID, Merchant ID, and others.

Results

The RandomForestClassifier model was trained with an accuracy of 99% on the test data, with recall and precision scores optimized for fraud detection.

Future Work:

Potential enhancements include:

    Incorporating more advanced models such as neural networks.
    Adding more robust features for transaction tracking and fraud detection.
    Deploying the project on cloud platforms.
