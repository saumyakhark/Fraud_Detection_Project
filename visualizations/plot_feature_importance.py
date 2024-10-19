# plot_feature_importance.py

import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = joblib.load('saved_models/fraud_detection_model.pkl')

# Load the preprocessed data
X_train_scaled = np.load('prepro/X_train_scaled.npy')

# Debugging: Print shape of the features
print("Shape of X_train_scaled:", X_train_scaled.shape)

# Replace these with the actual names of your features
feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12','feature13','feature14','feature15','feature16','feature17','feature18','feature19','feature20','feature21','feature22','feature23','feature24','feature25','feature26','feature27','feature28','feature29','feature30']  # Adjust this list

# Get feature importance from the model coefficients
importance = np.abs(model.coef_[0])

# Debugging prints
print("Model Coefficients:", model.coef_)
print("Number of Features in Data:", X_train_scaled.shape[1])
print("Number of Features in feature_names:", len(feature_names))
print("Importance Values:", importance)
print("Importance Length:", len(importance))

# Ensure feature_names and importance lengths match
if len(feature_names) != len(importance):
    raise ValueError("Length of feature_names does not match the length of importance values.")

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance based on Logistic Regression Coefficients')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.tight_layout()
plt.savefig('feature_importance.png')  # Save the plot as an image
plt.show()
