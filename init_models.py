import numpy as np

# Initialize random weights for 7 features
weights = np.random.uniform(-1, 1, size=(7,))
bias = np.random.uniform(-1, 1)

# Save them cleanly with no pickle dependencies
np.save("models/weights.npy", weights)
np.save("models/bias.npy", bias)

print("Weights and bias saved successfully.")
