import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model, save_model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and normalize preprocessed data
x_train = np.load('x_train.npy') / 255.0  # Shape: (num_samples, 32, 12, 3)
y_train = np.load('y_train.npy')          # Shape: (num_samples, 2)
x_test = np.load('x_test.npy') / 255.0    # Shape: (num_samples, 32, 12, 3)
y_test = np.load('y_test.npy')            # Shape: (num_samples, 2)

# Compute dynamic class weights
class_labels = np.argmax(y_train, axis=1)  # Convert one-hot to class indices
print(f"Class Distribution in y_train: {np.bincount(class_labels)}")  # Debugging: Check class distribution
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"Dynamic Class Weights: {class_weight_dict}")  # Debugging: Ensure weights are computed correctly

# Define the ResNet model
def create_res_net(input_shape=(32, 12, 3), num_filters=64, kernel_size=3, dropout_rate=0.5, l2_lambda=1e-4):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(5):  # Increased residual blocks
        residual = x
        if num_filters != residual.shape[-1]:
            residual = tf.keras.layers.Conv2D(num_filters, kernel_size=1, padding="same",
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(residual)
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same", activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.ReLU()(x)
        num_filters *= 2

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.9
        )
    )

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "AUC", "Precision", "Recall"]
    )
    return model

# Directory for saving model checkpoints
os.makedirs("model_checkpoints", exist_ok=True)

# Directory for saving visualizations
os.makedirs("visualizations", exist_ok=True)

# Train and save models
num_models = 2
fold_accuracies = []

for fold in range(1, num_models + 1):
    print(f"\nTraining Model {fold}...")

    # Create a new model instance
    model = create_res_net()
    checkpoint_path = f"model_checkpoints/axis_model_{fold}.h5"

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    # Train the model
    history = model.fit(
        x_train, y_train,  # Directly using the data without augmentation
        validation_data=(x_test, y_test),
        epochs=35,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Correctly passed class weights
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
    y_test_pred = np.argmax(best_model.predict(x_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)

    # Compute accuracy
    test_accuracy = np.mean(y_test_pred == y_test_true)
    fold_accuracies.append((test_accuracy, checkpoint_path))
    print(f"Model {fold} Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    report = classification_report(y_test_true, y_test_pred, target_names=["long axis", "short axis"])
    print(f"\nClassification Report for Model {fold}:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred)
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
    save_path = os.path.join(ensemble_dir, f"ensemble_model_{i}.h5")
    save_model(model, save_path)
    print(f"Saved Ensemble Model {i}: {save_path}")

# Ensemble predictions
ensemble_predictions = np.mean(
    [load_model(f"model_checkpoints/axis_model_{i}.h5").predict(x_test) for i in range(1, 6)], axis=0
)
final_predictions = np.argmax(ensemble_predictions, axis=1)
print("Final Ensemble Predictions Completed.")
# Function to plot training and validation metrics
def plot_training_metrics(history, fold, save_dir):
    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model {fold} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_dir}/loss_curve_model_{fold}.png", bbox_inches="tight")
    plt.close()
    
    # Plot Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model {fold} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_dir}/accuracy_curve_model_{fold}.png", bbox_inches="tight")
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# Function to plot classification metrics (precision, recall, f1-score)
def plot_metrics_bar_chart(y_true, y_pred, class_labels, title, save_path):
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        values = [report[cls][metric] for cls in class_labels]
        plt.figure()
        plt.bar(class_labels, values, color='skyblue')
        plt.title(f'{metric.capitalize()} - {title}')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        plt.savefig(f"{save_path}_{metric}.png", bbox_inches="tight")
        plt.close()

# Directory for saving visualizations
os.makedirs("visualizations", exist_ok=True)

# Class labels for the ResNet models
class_labels_resnet = ["long axis", "short axis"]

# Visualizations for each model
for fold in range(1, num_models + 1):
    print(f"Visualizing Model {fold}...")
    
    # Reload the best model for this fold
    checkpoint_path = f"model_checkpoints/axis_model_{fold}.h5"
    model = load_model(checkpoint_path)
    
    # Predict on test data
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    # Generate Confusion Matrix
    plot_confusion_matrix(
        y_true=y_test_true,
        y_pred=y_test_pred,
        class_labels=class_labels_resnet,
        title=f"Confusion Matrix - Model {fold}",
        save_path=f"visualizations/confusion_matrix_model_{fold}.png"
    )
    
    # Plot Metrics (Precision, Recall, F1-score)
    plot_metrics_bar_chart(
        y_true=y_test_true,
        y_pred=y_test_pred,
        class_labels=class_labels_resnet,
        title=f"Model {fold}",
        save_path=f"visualizations/model_{fold}_metrics"
    )
    
    # Training History Visualization
    history_path = f"visualizations/loss_curve_model_{fold}.png"
    print(f"Training metrics visualizations saved for Model {fold}")

# Visualize Ensemble Performance
ensemble_predictions = np.mean(
    [load_model(f"model_checkpoints/axis_model_{i}.h5").predict(x_test) for i in range(1, num_models + 1)], axis=0
)
final_predictions = np.argmax(ensemble_predictions, axis=1)

# Confusion Matrix for Ensemble
plot_confusion_matrix(
    y_true=y_test_true,
    y_pred=final_predictions,
    class_labels=class_labels_resnet,
    title="Confusion Matrix - Ensemble",
    save_path="visualizations/confusion_matrix_ensemble.png"
)

# Classification Metrics for Ensemble
plot_metrics_bar_chart(
    y_true=y_test_true,
    y_pred=final_predictions,
    class_labels=class_labels_resnet,
    title="Ensemble Model",
    save_path="visualizations/ensemble_metrics"
)

# Print Ensemble Performance
ensemble_accuracy = accuracy_score(y_test_true, final_predictions)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")