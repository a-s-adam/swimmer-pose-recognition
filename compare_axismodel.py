import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
x_test = np.load('x_test.npy') / 255.0    # Shape: (num_samples, 32, 12, 3)
y_test = np.load('y_test.npy')            # Shape: (num_samples, 2)

# Create separate versions of test data for LSTM and ResNet
# ResNet expects (num_samples, 32, 12, 3), no changes needed
x_test_resnet = x_test

# LSTM expects (num_samples, 32, 24), reshape accordingly
if x_test.shape[-1] == 3:
    x_test_lstm = x_test[:, :, :, :2].reshape(x_test.shape[0], x_test.shape[1], 24)  # Use first two channels for LSTM
else:
    raise ValueError(f"Unexpected last dimension of x_test: {x_test.shape[-1]}, expected 3.")

# Load the best LSTM and ResNet models
lstm_model_path = 'model_ensemble/lstm_model_1.h5'  # Path to the best LSTM model checkpoint
resnet_model_path = 'model_ensemble/axis_model_5.h5'  # Path to the best ResNet model checkpoint
#axis 1 62.6%
#2 74%
#3 72%
#4 67%
#5 62%


lstm_model = tf.keras.models.load_model(lstm_model_path)
resnet_model = tf.keras.models.load_model(resnet_model_path)

# Evaluate LSTM model
y_pred_lstm = np.argmax(lstm_model.predict(x_test_lstm), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nLSTM Model Performance:")
lstm_report = classification_report(y_true, y_pred_lstm, target_names=["long axis", "short axis"], digits=4)
print(lstm_report)

# Confusion matrix for LSTM
cm_lstm = confusion_matrix(y_true, y_pred_lstm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lstm, annot=True, fmt="d", cmap="Blues", xticklabels=["long axis", "short axis"], yticklabels=["long axis", "short axis"])
plt.title("Confusion Matrix - LSTM Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluate ResNet model
y_pred_resnet = np.argmax(resnet_model.predict(x_test_resnet), axis=1)

print("\nResNet Model Performance:")
resnet_report = classification_report(y_true, y_pred_resnet, target_names=["long axis", "short axis"], digits=4)
print(resnet_report)

# Confusion matrix for ResNet
cm_resnet = confusion_matrix(y_true, y_pred_resnet)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_resnet, annot=True, fmt="d", cmap="Blues", xticklabels=["long axis", "short axis"], yticklabels=["long axis", "short axis"])
plt.title("Confusion Matrix - ResNet Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Compare accuracy, precision, recall, and AUC for both models
lstm_accuracy = np.mean(y_pred_lstm == y_true)
resnet_accuracy = np.mean(y_pred_resnet == y_true)

print(f"\nComparison of Test Accuracies:")
print(f"LSTM Model Test Accuracy: {lstm_accuracy:.4f}")
print(f"ResNet Model Test Accuracy: {resnet_accuracy:.4f}")
