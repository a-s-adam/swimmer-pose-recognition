import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Fixed seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load processed skeletal data for training
FreeSkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\FreeSkel(c32).npy")
BackSkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\BackSkel(c32).npy")

# Load corresponding labels
FreeLabel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\FreeLabel(c32).npy")
BackLabel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\BackLabel(c32).npy")

# Combine Freestyle and Backstroke data for training
x_data = np.concatenate((FreeSkel, BackSkel), axis=0)
y_data = np.concatenate((FreeLabel, BackLabel), axis=0)

# Reshape data: Flatten last two dimensions to (sequence_length, num_keypoints * num_dimensions)
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], -1)

# Shuffle the data
indices = np.arange(x_data.shape[0])
np.random.shuffle(indices)
x_data = x_data[indices]
y_data = y_data[indices]

# Split into training (80%) and testing (20%)
split_index = int(x_data.shape[0] * 0.8)
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# One-hot encode the binary labels (Freestyle=0, Backstroke=1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Define the LSTM model with parameter flexibility
def create_lstm_model(input_shape, lstm_units, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(2, activation='softmax')  # Binary classification (Freestyle=0, Backstroke=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Configurations for diversity
configurations = [
    {"lstm_units": 128, "dropout_rate": 0.2, "learning_rate": 1e-3},
    {"lstm_units": 64, "dropout_rate": 0.3, "learning_rate": 5e-4},
    {"lstm_units": 256, "dropout_rate": 0.2, "learning_rate": 1e-4},
    {"lstm_units": 128, "dropout_rate": 0.1, "learning_rate": 1e-3},
    {"lstm_units": 96, "dropout_rate": 0.25, "learning_rate": 2e-4}
]

# Train 5 diverse LSTM models
long_axis_models = []  # Renamed variable for clarity
for i, config in enumerate(configurations):
    print(f"Training Long Axis Model {i+1} with config: {config}...")
    model = create_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]), 
                              lstm_units=config["lstm_units"], 
                              dropout_rate=config["dropout_rate"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)
    model.save(f"long_axis_model_{i+1}.h5")  # Save models with updated naming
    long_axis_models.append(model)

# Create a list of model paths
long_axis_models_paths = [f"long_axis_model_{i+1}.h5" for i in range(5)]

# Load all Long Axis models into a list
long_axis_models = [tf.keras.models.load_model(path) for path in long_axis_models_paths]

# Perform soft voting with all models
def soft_voting(models, x_data):
    probabilities = np.mean([model.predict(x_data) for model in models], axis=0)
    return np.argmax(probabilities, axis=1)

# Perform soft voting on the test set
ensemble_predictions = soft_voting(long_axis_models, x_test)

# Evaluate ensemble performance
y_test_classes = np.argmax(y_test, axis=1)
print("Classification Report:")
print(classification_report(y_test_classes, ensemble_predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, ensemble_predictions))

accuracy = accuracy_score(y_test_classes, ensemble_predictions)
print(f"Soft Voting Ensemble Accuracy: {accuracy}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Generate Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

# Generate Bar Chart for Metrics
def plot_metrics_bar_chart(y_true, y_pred, class_labels, title, save_path):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    values = {metric: [report[str(cls)][metric] for cls in range(len(class_labels))] for metric in metrics_to_plot}

    for metric, scores in values.items():
        plt.figure(figsize=(8, 5))
        plt.bar(class_labels, scores, color='skyblue')
        plt.title(f'{metric.capitalize()} for Each Class - {title}')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        for i, v in enumerate(scores):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        plt.savefig(f"{save_path}_{metric}.png", bbox_inches="tight")
        plt.close()
        print(f"{metric.capitalize()} bar chart saved to {save_path}_{metric}.png")

# Generate Accuracy Chart
def plot_accuracy_chart(models, x_test, y_test, title, save_path):
    accuracies = []
    for i, model in enumerate(models):
        predictions = np.argmax(model.predict(x_test), axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(true_classes, predictions)
        accuracies.append(accuracy)

    # Plot model-wise accuracies
    plt.figure(figsize=(8, 5))
    plt.bar([f'Model {i+1}' for i in range(len(models))], accuracies, color='skyblue')
    plt.title(f'Accuracy of Individual Models - {title}')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Accuracy chart saved to {save_path}")

# Visualize Performance of Long Axis Models
# Confusion Matrix
class_labels_long_axis = ["Freestyle", "Backstroke"]
plot_confusion_matrix(
    y_true=y_test_classes,
    y_pred=ensemble_predictions,
    class_labels=class_labels_long_axis,
    title="Confusion Matrix - Long Axis Models",
    save_path="visualizations/long_axis_confusion_matrix.png"
)

# Classification Metrics
plot_metrics_bar_chart(
    y_true=y_test_classes,
    y_pred=ensemble_predictions,
    class_labels=class_labels_long_axis,
    title="Long Axis Models",
    save_path="visualizations/long_axis_metrics"
)

# Accuracy for Individual Models
plot_accuracy_chart(
    models=long_axis_models,
    x_test=x_test,
    y_test=y_test,
    title="Long Axis Models",
    save_path="visualizations/long_axis_model_accuracies.png"
)
