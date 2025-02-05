import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
def create_sequences(skeleton_data, labels, frame_size=32, step=16):
    num_samples = (skeleton_data.shape[0] - frame_size) // step + 1
    sequences = np.empty((num_samples, frame_size, skeleton_data.shape[1], 3))  # Add z=0
    seq_labels = np.empty((num_samples,), dtype=int)  # Ensure labels are integers
    
    for i in range(num_samples):
        start = i * step
        end = start + frame_size
        sequences[i, :, :, :2] = skeleton_data[start:end]  # Copy x, y coordinates
        sequences[i, :, :, 2] = 0  # Add z=0
        seq_labels[i] = int(round(median(labels[start:end].flatten())))  # Round and convert to int
    
    return sequences, seq_labels

x_data, y_data = create_sequences(all_skel, all_labels)

# Visualize the original skeletal data
def visualize_skeletal_data(data, labels, title="Original Data"):
    print("Visualizing Skeletal Data")
    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(15, 5))
    for i, label in enumerate(unique_labels):
        sample = data[np.where(labels == label)[0][0]]  # First sample for the label
        ax = axes[i]
        ax.scatter(sample[:, 0], sample[:, 1], c='blue')
        ax.set_title(f"Class {int(label)}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize original data
visualize_skeletal_data(x_data[:, 0, :, :2], y_data, title="Original Data")

# Oversampling: Ensure `y_data` is discrete
x_data_flat = x_data.reshape(x_data.shape[0], -
