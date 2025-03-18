# this is for the hog svm classifier with different dataset sizes

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

# Set random seed for reproducibility
random_seed = 722 
random.seed(random_seed)
np.random.seed(random_seed)

# Path to your dataset
dataset_path = "NJZ_cropped_faces_dataset"

# Define the members
members = ["Danielle", "Hanni", "Minji", "Hyein", "Haerin"]

# Dataset sizes to experiment with (images per member)
dataset_sizes = [50, 100, 150, 180]

# Function to load data with a specified size per member
def load_data(size_per_member):
    images = []
    labels = []
    file_paths = []
    
    for idx, member in enumerate(members):
        member_path = os.path.join(dataset_path, member)
        image_paths = glob(os.path.join(member_path, "*.jpg"))
        
        # Randomly select the specified number of images per member
        if len(image_paths) > size_per_member:
            image_paths = random.sample(image_paths, size_per_member)
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64,64))
            images.append(img)
            labels.append(idx)
            file_paths.append(img_path)
    
    return np.array(images), np.array(labels), np.array(file_paths)

# Function to extract HOG features
def extract_hog_features(images):
    print("Extracting HOG features...")
    hog_features = []
    for image in images:
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Store results
results = {}

# Perform experiment for each dataset size
for size in dataset_sizes:
    print(f"\n=== Experiment with {size} images per member (Total: {size * len(members)} images) ===")
    
    # Load data for this size
    images, labels, file_paths = load_data(size)
    
    # Extract HOG features
    hog_features = extract_hog_features(images)
    
    # Perform cross-validation
    print("Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    svm = SVC(kernel='linear', probability=True, random_state=random_seed)
    
    # Get cross-validation accuracy scores
    cv_scores = cross_val_score(svm, hog_features, labels, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Get predictions for detailed metrics
    y_pred = cross_val_predict(svm, hog_features, labels, cv=cv)
    
    # Compute and print classification report
    print("\nClassification Report:")
    report = classification_report(labels, y_pred, target_names=members, output_dict=True)
    print(classification_report(labels, y_pred, target_names=members))
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Store results
    results[size] = {
        'accuracy': cv_scores.mean(),
        'std': cv_scores.std(),
        'precision': [report[member]['precision'] for member in members],
        'recall': [report[member]['recall'] for member in members],
        'f1': [report[member]['f1-score'] for member in members],
        'confusion_matrix': cm
    }
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (HOG+SVM, {size} images per member)")
    plt.colorbar()
    tick_marks = np.arange(len(members))
    plt.xticks(tick_marks, members, rotation=45)
    plt.yticks(tick_marks, members)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Summarize results
print("\n=== Summary of Results ===")
for size in dataset_sizes:
    print(f"\nDataset Size: {size} images per member (Total: {size * len(members)})")
    print(f"Mean CV Accuracy: {results[size]['accuracy']:.4f} ± {results[size]['std']:.4f}")
    print("Per-class Precision:", [f"{x:.4f}" for x in results[size]['precision']])
    print("Per-class Recall:", [f"{x:.4f}" for x in results[size]['recall']])
    print("Per-class F1-score:", [f"{x:.4f}" for x in results[size]['f1']])

# Plot accuracy vs. dataset size
plt.figure(figsize=(8, 5))
accuracies = [results[size]['accuracy'] for size in dataset_sizes]
stds = [results[size]['std'] for size in dataset_sizes]
plt.errorbar(dataset_sizes, accuracies, yerr=stds, fmt='-o', capsize=5)
plt.title("Mean CV Accuracy vs. Dataset Size (HOG+SVM)")
plt.xlabel("Images per Member")
plt.ylabel("Mean CV Accuracy")
plt.grid(True)
plt.show()