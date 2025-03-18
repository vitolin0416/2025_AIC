# this is for the hog svm classifier with pca

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
from glob import glob
import random   

# Set random seed for reproducibility
random_seed = 722
random.seed(random_seed)
np.random.seed(random_seed)

# Path to your dataset
dataset_path = "NJZ_cropped_faces_dataset"

# Define the members
members = ["Danielle", "Hanni", "Minji", "Hyein", "Haerin"]

# PCA components to test
# pca_components = [50, 100, 200, 400, 800, 1600, 3200, None]  # Expanded list
pca_components = [20, 50, 100, 200, 400, 600, 800, 1000, 1200, None]
# Target size for oversampling (based on largest class, e.g., Minji/Haerin)
target_size_per_member = 300 

# Store images and labels
images = []
labels = []
file_paths = []

# Load all available images for each member
for idx, member in enumerate(members):
    member_path = os.path.join(dataset_path, member)
    image_paths = glob(os.path.join(member_path, "*.jpg"))
    
    # Use all available images (no cap)
    print(f"Loaded {len(image_paths)} images for {member}")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))  
        images.append(img)
        labels.append(idx)
        file_paths.append(img_path)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)
file_paths = np.array(file_paths)

# Extract HOG features
print("\nExtracting HOG features...")
hog_features = []
for image in images:
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    hog_features.append(features)

hog_features = np.array(hog_features)
print(f"Original HOG feature dimensionality: {hog_features.shape[1]}")

# Apply SMOTE to balance the dataset
print(f"\nApplying SMOTE to oversample to {target_size_per_member} images per member...")
smote = SMOTE(sampling_strategy={i: target_size_per_member for i in range(len(members))},
              random_state=random_seed)
hog_features_smote, labels_smote = smote.fit_resample(hog_features, labels)

# Check new dataset size
print(f"New dataset size after SMOTE: {len(labels_smote)} samples")
unique, counts = np.unique(labels_smote, return_counts=True)
for member_idx, count in zip(unique, counts):
    print(f"{members[member_idx]}: {count} samples")

# Store results
results = {}

# Perform experiment for each PCA setting
for n_components in pca_components:
    if n_components is None:
        print(f"\n=== Experiment with full HOG features (no PCA, {hog_features_smote.shape[1]} components) ===")
        features = hog_features_smote
    else:
        print(f"\n=== Experiment with PCA ({n_components} components) ===")
        pca = PCA(n_components=n_components, random_state=random_seed)
        features = pca.fit_transform(hog_features_smote)
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Perform cross-validation
    print("Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    svm = SVC(kernel='linear', probability=True, random_state=random_seed)
    
    # Get cross-validation accuracy scores
    cv_scores = cross_val_score(svm, features, labels_smote, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Get predictions for detailed metrics (optional, but not plotted)
    y_pred = cross_val_predict(svm, features, labels_smote, cv=cv)
    
    # Compute and print classification report (optional, for reference)
    print("\nClassification Report:")
    print(classification_report(labels_smote, y_pred, target_names=members))
    
    # Store results
    results[n_components if n_components is not None else 'full'] = {
        'accuracy': cv_scores.mean(),
        'std': cv_scores.std()
    }

# Summarize results
print("\n=== Summary of Results ===")
for n_components in results:
    print(f"\nPCA Components: {n_components}")
    print(f"Mean CV Accuracy: {results[n_components]['accuracy']:.4f} ± {results[n_components]['std']:.4f}")

# Plot accuracy vs. PCA components
plt.figure(figsize=(8, 5))
components_list = [n if n is not None else hog_features_smote.shape[1] for n in pca_components]
accuracies = [results[n if n is not None else 'full']['accuracy'] for n in pca_components]
stds = [results[n if n is not None else 'full']['std'] for n in pca_components]
plt.errorbar(components_list, accuracies, yerr=stds, fmt='-o', capsize=5)
plt.title("Mean CV Accuracy vs. PCA Components (HOG+SVM, 64x64 with SMOTE)")
plt.xlabel("Number of PCA Components")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)
plt.xticks(components_list, [str(n) if n is not None else 'Full' for n in pca_components], rotation=45)
plt.tight_layout()
plt.show()