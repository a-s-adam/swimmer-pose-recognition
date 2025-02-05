import numpy as np
import tensorflow as tf
from statistics import median

# Load skeletal training data for different swimming styles
skel0 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\freestyle-training.npy")  # Freestyle
skel1 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\back-training.npy")      # Backstroke
skel2 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\fly-training.npy")       # Butterfly
skel3 = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\breast-training.npy")    # Breaststroke

# Create labels: Freestyle (class 0), Backstroke (class 1), Butterfly (class 2), Breaststroke (class 3)
label0 = np.zeros((skel0.shape[0], 1))  # Freestyle labeled as 0
label1 = np.ones((skel1.shape[0], 1))   # Backstroke labeled as 1
label2 = np.full((skel2.shape[0], 1), 2)  # Butterfly labeled as 2
label3 = np.full((skel3.shape[0], 1), 3)  # Breaststroke labeled as 3

# Combine all skeletons and labels
all_skel = np.concatenate((skel0, skel1, skel2, skel3), axis=0)
all_labels = np.concatenate((label0, label1, label2, label3), axis=0)

# Shuffle the data
indices = np.arange(all_skel.shape[0])
np.random.shuffle(indices)
all_skel = all_skel[indices]
all_labels = all_labels[indices]

# Reshape into sequences of 32 frames
def create_sequences(skeleton_data, labels, frame_size=32, step=16):
    num_samples = (skeleton_data.shape[0] - frame_size) // step + 1
    sequences = np.empty((num_samples, frame_size, skeleton_data.shape[1], 3))  # Add z=0
    seq_labels = np.empty((num_samples,))
    
    for i in range(num_samples):
        start = i * step
        end = start + frame_size
        sequences[i, :, :, :2] = skeleton_data[start:end]  # Copy x, y coordinates
        sequences[i, :, :, 2] = 0  # Add z=0
        seq_labels[i] = median(labels[start:end].flatten())  # Majority label for sequence
    
    return sequences, seq_labels

x_data, y_data = create_sequences(all_skel, all_labels)

# Split into training (80%) and testing (20%)
split_index = int(x_data.shape[0] * 0.8)
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

# One-hot encode the four-class labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- Data Augmentation Functions ---
def inject_noise(coordinates, mean=0, std=0.3):
    """Add Gaussian noise to skeletal data."""
    noise = np.random.normal(mean, std, coordinates.shape)
    noisy_coordinates = coordinates + noise
    return noisy_coordinates

def rotate_augment(coordinates, max_angle=45):
    """Randomly rotate skeleton data around the center for x, y coordinates only."""
    angles = np.random.uniform(low=-max_angle, high=max_angle)
    center = np.mean(coordinates[:, :, :, :2], axis=(1, 2), keepdims=True)  # Compute the center of rotation for x, y only

    # Rotation matrix for 2D rotation (x, y)
    rotation_matrix = np.array([[np.cos(np.radians(angles)), -np.sin(np.radians(angles))],
                                 [np.sin(np.radians(angles)), np.cos(np.radians(angles))]])

    rotated_coordinates = coordinates.copy()  # Copy original coordinates
    for i in range(coordinates.shape[0]):  # Iterate over sequences
        for j in range(coordinates.shape[1]):  # Iterate over frames
            translated = coordinates[i, j, :, :2] - center[i, 0]  # Translate x, y to the center
            rotated = np.dot(rotation_matrix, translated.T).T  # Apply rotation
            rotated_coordinates[i, j, :, :2] = rotated + center[i, 0]  # Translate back

    return rotated_coordinates

def mirror_augment(coordinates):
    """Mirror skeletal data horizontally by flipping x-coordinates."""
    mirrored_coordinates = coordinates.copy()
    mirrored_coordinates[:, :, :, 0] = -mirrored_coordinates[:, :, :, 0]
    return mirrored_coordinates

# Apply Data Augmentation
# 1. Noise Augmentation
noisy_x_train = inject_noise(x_train)

# 2. Rotation Augmentation
rotated_x_train = rotate_augment(x_train)

# 3. Mirror Augmentation
mirrored_x_train = mirror_augment(x_train)

# Combine Original and Augmented Data
x_train_augmented = np.concatenate((x_train, noisy_x_train, rotated_x_train, mirrored_x_train), axis=0)
y_train_augmented = np.concatenate((y_train, y_train, y_train, y_train), axis=0)

print(f"Augmented x_train shape: {x_train_augmented.shape}")
print(f"Augmented y_train shape: {y_train_augmented.shape}")

# Save the augmented training and testing datasets
np.save("x_train.npy", x_train_augmented)
np.save("y_train.npy", y_train_augmented)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)

print(f"Saved x_train.npy, y_train.npy, x_test.npy, y_test.npy")
