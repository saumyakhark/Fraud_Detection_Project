from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('saved_models/fraud_detection_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        transaction_amount = float(request.form['Transaction Amount'])
        transaction_time = float(request.form['Transaction Time'])
        user_id = int(request.form['User ID'])
        merchant_id = int(request.form['Merchant ID'])
        location_code = int(request.form['Location Code'])
        transaction_type = int(request.form['Transaction Type'])
        account_age = int(request.form['Account Age (days)'])
        num_previous_transactions = int(request.form['Number of Previous Transactions'])
        avg_transaction_amount = float(request.form['Average Transaction Amount'])
        suspicious_activity_count = int(request.form['Suspicious Activity Count'])

        # Prepare features array for prediction
        features = np.array([[transaction_amount, transaction_time, user_id, merchant_id, location_code,
                              transaction_type, account_age, num_previous_transactions,
                              avg_transaction_amount, suspicious_activity_count]])

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction to user-friendly message
        result = "Legitimate Transaction" if prediction[0] == 0 else "Fraudulent Transaction"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
