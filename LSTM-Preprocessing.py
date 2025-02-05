import numpy as np
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from statistics import median
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
np.random.seed(42)
tf.random.set_seed(42)

# Load skeletal training data for different swimming styles
skel0 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\freestyle-training.npy")  # Freestyle
skel1 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\back-training.npy")      # Backstroke
skel2 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\fly-training.npy")       # Butterfly
skel3 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\breast-training.npy")    # Breaststroke

# Create labels: Freestyle and Backstroke (class 0), Butterfly and Breaststroke (class 1)
label0 = np.zeros((skel0.shape[0], 1))  # Freestyle labeled as 0
label1 = np.zeros((skel1.shape[0], 1))  # Backstroke labeled as 0
label2 = np.ones((skel2.shape[0], 1))   # Butterfly labeled as 1
label3 = np.ones((skel3.shape[0], 1))   # Breaststroke labeled as 1

# Combine all skeletons and labels
all_skel = np.concatenate((skel0, skel1, skel2, skel3), axis=0)
all_labels = np.concatenate((label0, label1, label2, label3), axis=0)

# Shuffle the data
indices = np.arange(all_skel.shape[0])
np.random.shuffle(indices)
all_skel = all_skel[indices]
all_labels = all_labels[indices]

# Reshape into sequences of 32 frames
def create_sequences(skeleton_data, labels, frame_size=32, step=8):
    num_samples = (skeleton_data.shape[0] - frame_size) // step + 1
    sequences = np.empty((num_samples, frame_size, skeleton_data.shape[1], skeleton_data.shape[2]))  # Maintain joint and coord structure
    seq_labels = np.empty((num_samples,), dtype=int)  # Ensure labels are integers
    
    for i in range(num_samples):
        start = i * step
        end = start + frame_size
        sequences[i] = skeleton_data[start:end]
        seq_labels[i] = int(round(median(labels[start:end].flatten())))  # Round and convert to int
    
    return sequences, seq_labels

x_data, y_data = create_sequences(all_skel, all_labels)

# Oversampling: Ensure `y_data` is discrete
x_data_flat = x_data.reshape(x_data.shape[0], -1)  # Flatten for oversampling
ros = RandomOverSampler(random_state=42)
x_data_resampled, y_data_resampled = ros.fit_resample(x_data_flat, y_data)  # Now y_data is discrete

# Reshape back to original dimensions
x_data_resampled = x_data_resampled.reshape(-1, 32, x_data.shape[2], x_data.shape[3])
y_data_resampled = tf.keras.utils.to_categorical(y_data_resampled, num_classes=2)

# Perform a stratified split to preserve class distribution
x_train, x_test, y_train, y_test = train_test_split(
    x_data_resampled, y_data_resampled,
    test_size=0.2,  # 20% for testing
    stratify=y_data_resampled,  # Preserve class distribution
    random_state=42
)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Save the preprocessed data for model training
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print("Saved x_train.npy, y_train.npy, x_test.npy, y_test.npy")
