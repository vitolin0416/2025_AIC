# this is for the kmeans clustering

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from skimage.feature import hog

# Set random seed for reproducibility
random_seed = 42
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
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

# Perform K-Means clustering
print("Performing K-Means clustering...")
kmeans = KMeans(n_clusters=5, random_state=random_seed, n_init=10)
cluster_labels = kmeans.fit_predict(hog_features)

# Evaluation metrics
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)

print(f"\nEvaluation Metrics:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Compute initial confusion matrix (before alignment)
cm_initial = confusion_matrix(labels, cluster_labels)
print("\nInitial Confusion Matrix (True Labels vs. Cluster Labels, Unaligned):")
print(cm_initial)

# Align cluster labels with true labels using the Hungarian algorithm
def align_labels(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize diagonal by minimizing -cm
    label_mapping = {old_label: new_label for old_label, new_label in zip(col_ind, row_ind)}
    aligned_labels = np.array([label_mapping[label] for label in cluster_labels])
    return aligned_labels

# Get aligned cluster labels
aligned_cluster_labels = align_labels(labels, cluster_labels)

# Compute aligned confusion matrix
cm_aligned = confusion_matrix(labels, aligned_cluster_labels)
print("\nAligned Confusion Matrix (True Labels vs. Aligned Cluster Labels):")
print(cm_aligned)

# Compute clustering accuracy based on aligned labels
clustering_accuracy = np.sum(np.diag(cm_aligned)) / np.sum(cm_aligned)
print(f"\nClustering Accuracy (after alignment): {clustering_accuracy:.4f}")

# Visualize the aligned confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm_aligned, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Aligned Confusion Matrix (K-Means Clustering)")
plt.colorbar()
tick_marks = np.arange(len(members))
plt.xticks(tick_marks, members, rotation=45)
plt.yticks(tick_marks, members)
plt.ylabel('True Label')
plt.xlabel('Aligned Cluster Label')
plt.tight_layout()
plt.show()

# Analyze cluster composition
print("\nCluster Composition (Number of samples per cluster):")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} samples")