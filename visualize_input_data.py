import numpy as np
import matplotlib.pyplot as plt

# Load processed skeletal data for training
FreeSkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\FreeSkel(c32).npy")
BackSkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\BackSkel(c32).npy")
FlySkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\FlySkel(c32).npy")
BreastSkel = np.load("C:\\Users\\austi\\Documents\\EE299\\yolov7\\BreastSkel(c32).npy")

# Combine data into a list for visualization
all_skel = [FreeSkel, BackSkel, FlySkel, BreastSkel]
labels = ['Freestyle', 'Backstroke', 'Butterfly', 'Breaststroke']

# Visualize 1 skeletal data sample per class
def visualize_single_sample(data, labels):
    print("Visualizing One Sample Per Class")
    fig, axes = plt.subplots(1, len(data), figsize=(15, 5))
    for i, skel_data in enumerate(data):
        try:
            sample = skel_data[0]  # Select the first sample
            ax = axes[i]
            # Ensure the data has x, y coordinates for visualization
            if sample.ndim >= 2:
                ax.scatter(sample[:, 0], sample[:, 1], c='blue')
                ax.set_title(f"{labels[i]} - Sample 1")
                ax.set_xlim(-1, 1)  # Adjust according to data range
                ax.set_ylim(-1, 1)  # Adjust according to data range
                ax.set_xlabel("X-axis")
                ax.set_ylabel("Y-axis")
                ax.grid(True)
            else:
                ax.set_title(f"Invalid Data for {labels[i]}")
        except Exception as e:
            ax.set_title(f"Error: {e}")
    plt.suptitle("Skeletal Data - 1 Sample Per Class", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_single_sample(all_skel, labels)
