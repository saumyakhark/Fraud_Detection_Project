# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create directory for saving preprocessed data if it doesn't exist
os.makedirs('preprocessed_data', exist_ok=True)

# Load your dataset
data = pd.read_csv('data/creditcard.csv')

# Assuming your target variable is in a column named 'Class' and features are in other columns
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed data as NumPy files
np.save('prepro/X_train_scaled.npy', X_train_scaled)
np.save('prepro/X_test_scaled.npy', X_test_scaled)
np.save('prepro/y_train.npy', y_train)
np.save('prepro/y_test.npy', y_test)

print("Preprocessing complete. Data saved.")
