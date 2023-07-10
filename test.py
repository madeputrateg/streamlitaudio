import numpy as np

# Define your target labels
labels = np.array([1, 2, 0, 2])  # Example labels

# Determine the number of classes in your dataset
num_classes = np.max(labels) + 1

# Perform one-hot encoding
one_hot = np.eye(num_classes)[labels]

# Print the one-hot encoded array
print(one_hot)