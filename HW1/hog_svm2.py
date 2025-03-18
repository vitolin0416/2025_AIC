# this is for the hog svm classifier with 64x64 images

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from glob import glob
import random
import shutil

# Set random seed for reproducibility
random_seed = 722
random.seed(random_seed)
np.random.seed(random_seed)

# Path to your dataset
dataset_path = "NJZ_cropped_faces_dataset"

# Define the members
members = ["Danielle", "Hanni", "Minji", "Hyein", "Haerin"]

# Store images and labels
images = []
labels = []
file_paths = []

# Load images for each member
for idx, member in enumerate(members):
    member_path = os.path.join(dataset_path, member)
    image_paths = glob(os.path.join(member_path, "*.jpg"))
    
    # Randomly select 180 images per member
    if len(image_paths) > 180:
        image_paths = random.sample(image_paths, 180)
    
    for img_path in image_paths:
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (64, 64))  # Resize to consistent dimensions
        
        # Append image, label, and file path
        images.append(img)
        labels.append(idx)
        file_paths.append(img_path)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)
file_paths = np.array(file_paths)

# Extract HOG features
print("Extracting HOG features...")
hog_features = []
for image in images:
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    hog_features.append(features)

hog_features = np.array(hog_features)

# Perform cross-validation
print("Performing 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
svm = SVC(kernel='linear', probability=True, random_state=random_seed)

# Cross-validation scores (accuracy)
cv_scores = cross_val_score(svm, hog_features, labels, cv=cv, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# After computing cv_scores
from sklearn.model_selection import cross_val_predict

# Get predictions across all folds
y_pred_all = cross_val_predict(svm, hog_features, labels, cv=cv)

# Overall classification report
print("\nOverall Classification Report:")
print(classification_report(labels, y_pred_all, target_names=members))

# Overall confusion matrix
overall_cm = confusion_matrix(labels, y_pred_all)
print("\nOverall Confusion Matrix:")
print(overall_cm)

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix (SVM)")
plt.colorbar()
tick_marks = np.arange(len(members))
plt.xticks(tick_marks, members, rotation=45)
plt.yticks(tick_marks, members)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()