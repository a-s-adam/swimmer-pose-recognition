import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model, save_model
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler

# Fixed seed
np.random.seed(42)
tf.random.set_seed(42)

# Load and normalize preprocessed data
x_train = np.load('x_train.npy') / 255.0  # Shape: (num_samples, 32, 12, 2)
y_train = np.load('y_train.npy')          # Shape: (num_samples, 2)
x_test = np.load('x_test.npy') / 255.0    # Shape: (num_samples, 32, 12, 2)
y_test = np.load('y_test.npy')            # Shape: (num_samples, 2)

# Reshape data for LSTM input
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], -1)  # Shape: (num_samples, 32, 24)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], -1)      # Shape: (num_samples, 32, 24)

# Convert one-hot encoded labels back to class labels for resampling
class_labels = np.argmax(y_train, axis=1)

# Address class imbalance by oversampling
ros = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = ros.fit_resample(x_train.reshape((x_train.shape[0], -1)), class_labels)
x_train_resampled = x_train_resampled.reshape((x_train_resampled.shape[0], 32, 24))
y_train_resampled = tf.keras.utils.to_categorical(y_train_resampled, num_classes=2)

# Compute dynamic class weights
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"Dynamic Class Weights: {class_weight_dict}")  # Debugging: Ensure weights are computed correctly

# Define the LSTM model
def create_lstm_model(input_shape, lstm_units, dropout_rate, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(lstm_units // 4, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# Directory for saving model checkpoints
os.makedirs("model_checkpoints", exist_ok=True)

# Directory for saving visualizations
os.makedirs("visualizations", exist_ok=True)

# Implement K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_resampled), 1):
    print(f"\nTraining Model {fold}...")
    
    # Split the data for the current fold
    x_train_fold, x_val_fold = x_train_resampled[train_idx], x_train_resampled[val_idx]
    y_train_fold, y_val_fold = y_train_resampled[train_idx], y_train_resampled[val_idx]
    
    # Create and compile the model
    model = create_lstm_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        lstm_units=64,  # Increased LSTM units
        dropout_rate=0.5,  # Increased dropout rate for better regularization
        num_classes=2
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Changed optimizer to Adam
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", "AUC", "Precision", "Recall"])

    # Callbacks
    checkpoint_path = f"model_checkpoints/lstm_model_{fold}.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)  # More aggressive learning rate reduction
    ]

    # Train the model
    history = model.fit(
        x_train_fold, y_train_fold,
        validation_data=(x_val_fold, y_val_fold),
        epochs=50,  # Increased epochs to 50
        batch_size=32,  # Reduced batch size for more frequent updates
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history plot
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"Loss Curve - Model {fold}")
    plt.savefig(f"visualizations/loss_curve_model_{fold}.png")
    plt.close()

    # Evaluate the best model
    best_model = load_model(checkpoint_path)
    y_val_pred = np.argmax(best_model.predict(x_val_fold), axis=1)
    y_val_true = np.argmax(y_val_fold, axis=1)

    # Compute accuracy
    val_accuracy = np.mean(y_val_pred == y_val_true)
    fold_accuracies.append((val_accuracy, checkpoint_path))
    print(f"Model {fold} Validation Accuracy: {val_accuracy:.4f}")

    # Classification report
    report = classification_report(y_val_true, y_val_pred, target_names=["long axis", "short axis"])
    print(f"\nClassification Report for Model {fold}:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_val_true, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["long axis", "short axis"], yticklabels=["long axis", "short axis"])
    plt.title(f"Confusion Matrix for Model {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"visualizations/confusion_matrix_model_{fold}.png")
    plt.close()

# Ensemble top 3 models
fold_accuracies.sort(reverse=True, key=lambda x: x[0])
ensemble_dir = "model_ensemble"
os.makedirs(ensemble_dir, exist_ok=True)
for i, (accuracy, model_path) in enumerate(fold_accuracies[:3], start=1):
    model = load_model(model_path)
    save_path = os.path.join(ensemble_dir, f"lstm_model_{i}.h5")
    save_model(model, save_path)
    print(f"Saved Ensemble Model {i}: {save_path}")

print("Final Ensemble Models Saved.")
