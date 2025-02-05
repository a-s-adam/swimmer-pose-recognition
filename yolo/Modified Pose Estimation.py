import numpy as np
import tensorflow as tf

# Assuming x_train is your training data
x_train = np.load("x_train.npy")

# Define the data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.1
)

# Fit the data generator (if required)
datagen.fit(x_train)

# Generate a batch of augmented images
augmented_iterator = datagen.flow(x_train, batch_size=1)

# Get the first augmented sample
augmented_sample = next(augmented_iterator)[0]

# Print the shape and values of the augmented sample
print("Shape of augmented sample:", augmented_sample.shape)
print("Augmented sample values:\n", augmented_sample)
