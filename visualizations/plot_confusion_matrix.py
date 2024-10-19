# plot_confusion_matrix.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np

# Load the model
model = joblib.load('saved_models/fraud_detection_model.pkl')

# Assuming you have test data (X_test and y_test)
X_test_scaled = np.load('prepro/X_test_scaled.npy')  # Load your test data
y_test = np.load('prepro/y_test.npy')  # Load the true labels

# Make predictions
y_pred = model.predict(X_test_scaled)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
