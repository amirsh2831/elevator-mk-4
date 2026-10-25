import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import os

# Load the features and labels you saved
X = np.load('features.npy')  # Shape: (100, 84)
y = np.load('labels.npy')   # Shape: (100,)

print(f"Loaded features: {X.shape}")
print(f"Loaded labels: {y.shape}")

# ====================== 1. Visualization: Average pattern per digit ======================
# We will plot the average energy, average (abs mean), and ZCR trajectory for each digit

fig, axes = plt.subplots(10, 3, figsize=(15, 25))
fig.suptitle('Average Feature Trajectories for Each Digit (After Resizing to 28 Frames)', fontsize=16)

for digit in range(10):
    # Samples for this digit: indices digit*10 to digit*10+9
    digit_indices = np.where(y == digit)[0]
    digit_features = X[digit_indices]  # (10, 84)
    
    # Reshape to (10, 28, 3)
    digit_reshaped = digit_features.reshape(10, 32, 3)
    
    # Mean across the 10 samples
    mean_trajectory = np.mean(digit_reshaped, axis=0)  # (28, 3)
    
    # Plot each feature
    frames = np.arange(32)
    axes[digit, 0].plot(frames, mean_trajectory[:, 0], color='blue')
    axes[digit, 0].set_title(f'Digit {digit+1} - Normalized Energy')
    axes[digit, 0].grid(alpha=0.3)
    
    axes[digit, 1].plot(frames, mean_trajectory[:, 1], color='orange')
    axes[digit, 1].set_title(f'Digit {digit+1} - Normalized Average (abs)')
    axes[digit, 1].grid(alpha=0.3)
    
    axes[digit, 2].plot(frames, mean_trajectory[:, 2], color='green')
    axes[digit, 2].set_title(f'Digit {digit+1} - Normalized ZCR')
    axes[digit, 2].grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# ====================== 2. Train/Test Split: Exactly 7 train + 3 test per digit ======================
# We do it manually to guarantee exactly 3 test samples per class

train_indices = []
test_indices = []

for digit in range(10):
    digit_indices = np.where(y == digit)[0]  # 10 indices
    # First 7 for train, last 3 for test (since files are ordered)
    train_indices.extend(digit_indices[:7])
    test_indices.extend(digit_indices[7:10])

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"\nTrain set: {X_train.shape[0]} samples (70)")
print(f"Test set: {X_test.shape[0]} samples (30)")

# ====================== 3. Feature Scaling (Standardization) ======================
# Very important for neural networks!

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)        # Apply same transformation to test

# Save everything for the next step (MLP training)
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Optional: save the scaler for later use (e.g., new recordings)
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("\nSaved scaled train/test sets and scaler.")
print("Visualization plot shown above.")
print("=== STEP 2 COMPLETE ===")