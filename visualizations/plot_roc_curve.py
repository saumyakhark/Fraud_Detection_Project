# plot_roc_curve.py

import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('../saved_models/fraud_detection_model.pkl')

# Load the test data
X_test_scaled = np.load('../prepro/X_test_scaled.npy')
y_test = np.load('../prepro/y_test.npy')

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line for random guess
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
