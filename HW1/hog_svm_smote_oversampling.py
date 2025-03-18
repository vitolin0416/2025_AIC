# this is for the hog svm classifier with smote oversampling

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
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
    
    # No cap; use all available images
    print(f"Loaded {len(image_paths)} images for {member}")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64,64))
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

# Perform cross-validation on SMOTE-balanced data
print("\nPerforming 5-fold cross-validation on SMOTE-balanced data...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
svm = SVC(kernel='linear', probability=True, random_state=random_seed)

# Get cross-validation accuracy scores
cv_scores = cross_val_score(svm, hog_features_smote, labels_smote, cv=cv, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Get predictions for detailed metrics
y_pred = cross_val_predict(svm, hog_features_smote, labels_smote, cv=cv)

# Compute and print classification report
print("\nClassification Report:")
report = classification_report(labels_smote, y_pred, target_names=members, output_dict=True)
print(classification_report(labels_smote, y_pred, target_names=members))

# Compute confusion matrix
cm = confusion_matrix(labels_smote, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (HOG+SVM with SMOTE)")
plt.colorbar()
tick_marks = np.arange(len(members))
plt.xticks(tick_marks, members, rotation=45)
plt.yticks(tick_marks, members)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Summarize per-class metrics
print("\nPer-class Metrics Summary:")
print("Precision:", [f"{report[member]['precision']:.4f}" for member in members])
print("Recall:", [f"{report[member]['recall']:.4f}" for member in members])
print("F1-score:", [f"{report[member]['f1-score']:.4f}" for member in members])