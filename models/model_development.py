import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv('data/creditcard.csv')

# Sample 10% of the data for faster testing
data_sample = data.sample(frac=0.1, random_state=42)  # Adjust fraction as needed

# Define your features and target variable
X = data_sample[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']]  # Use the relevant feature columns
y = data_sample['Class']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with fewer trees
model = RandomForestClassifier(n_estimators=10, random_state=42)  # Reduced number of trees
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'saved_models/fraud_detection_model.pkl')
joblib.dump(scaler, 'saved_models/scaler.pkl')

# Optional: Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)

# Print the classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
