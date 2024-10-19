import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "data/creditcard.csv"  # Replace with your actual dataset file path
data = pd.read_csv(file_path)

# Rename columns to match your desired feature names
data.columns = ['Transaction Time', 'Transaction Amount', 'user_id', 'merchant_id', 'location_code', 'transaction_type',
                'account_age', 'num_previous_transactions', 'avg_transaction_amount', 'suspicious_activity', 'Class']

# Dropping the 'Class' for feature selection and keeping it in y for labels
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the preprocessed datasets
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test_scaled.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("Preprocessing completed. The preprocessed datasets have been saved.")
