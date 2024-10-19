# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Step 1: Load the Data
# Ensure you download the Kaggle dataset and place it in the same directory as this script
# Dataset link: https://www.kaggle.com/mlg-ulb/creditcardfraud
df = pd.read_csv('data/creditcard.csv')

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Data Cleaning
# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# No missing values in this dataset, but here's a generic way to handle missing values if any:
# df.fillna(df.median(), inplace=True)

# Step 3: Feature Engineering
# Convert 'Time' feature into hours of the day
df['Hour'] = df['Time'].apply(lambda x: (x / 3600) % 24)

# Normalize the 'Amount' feature
df['Normalized_Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

# Drop original 'Time' and 'Amount' columns as we have created new features from them
df = df.drop(['Time', 'Amount'], axis=1)

# Step 4: Exploratory Data Analysis (EDA)
# Plot class distribution (Fraudulent vs Non-Fraudulent Transactions)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0 = Non-Fraud, 1 = Fraud)')
plt.show()

# Plot the distribution of normalized transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['Normalized_Amount'], bins=50, kde=True)
plt.title('Distribution of Normalized Transaction Amounts')
plt.show()

# Plot the number of transactions by hour of the day
plt.figure(figsize=(10, 6))
df['Hour'].hist(bins=24, edgecolor='k')
plt.title('Distribution of Transactions by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.show()

# Step 5: Handle Imbalanced Data using SMOTE
# Split the features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Use SMOTE to handle class imbalance by oversampling the minority class (fraudulent transactions)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the new class distribution after resampling
print(f"\nClass distribution after SMOTE resampling: {Counter(y_resampled)}")

# Step 6: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 7: Standardize the Data
scaler = StandardScaler()

# Scale the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display scaled data sample
print("\nFirst 5 rows of scaled training data:")
print(X_train_scaled[:5])

# Saving preprocessed data
preprocessed_data = {
    "X_train": X_train_scaled,
    "X_test": X_test_scaled,
    "y_train": y_train,
    "y_test": y_test
}

# Save to files (optional, if you'd like to persist the preprocessed data)
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nPreprocessing completed successfully!")
