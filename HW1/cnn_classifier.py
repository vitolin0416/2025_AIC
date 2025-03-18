# this is for the cnn classifier

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(idx)
        file_paths.append(img_path)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)
file_paths = np.array(file_paths)

# Normalize pixel values to 0-1
images = images / 255.0

# Function to create CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
cv_scores = []
all_y_true = []
all_y_pred = []

print("Performing 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(images, labels)):
    print(f"\nFold {fold+1}/5")
    
    # Split data
    X_train_fold, X_val_fold = images[train_idx], images[val_idx]
    y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]
    
    # Convert to one-hot encoding
    y_train_fold_onehot = to_categorical(y_train_fold, num_classes=5)
    y_val_fold_onehot = to_categorical(y_val_fold, num_classes=5)
    
    # Create data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and train model
    model = create_model()
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        datagen.flow(X_train_fold, y_train_fold_onehot, batch_size=32),
        epochs=15,
        validation_data=(X_val_fold, y_val_fold_onehot),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold_onehot, verbose=0)
    cv_scores.append(val_acc)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Get predictions for this fold
    y_pred_proba = model.predict(X_val_fold)
    y_pred = np.argmax(y_pred_proba, axis=1)  # Convert probabilities to class labels
    y_true = y_val_fold  # True labels (not one-hot encoded)
    
    # Store true and predicted labels for overall metrics
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)
    
    # Per-fold classification report
    print(f"\nClassification Report for Fold {fold+1}:")
    print(classification_report(y_true, y_pred, target_names=members))
    
    # Per-fold confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for Fold {fold+1}:\n{cm}")

# Print cross-validation results
print("\nCross-validation results:")
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.4f}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Overall classification report across all folds
print("\nOverall Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=members))

# Overall confusion matrix across all folds
overall_cm = confusion_matrix(all_y_true, all_y_pred)
print("\nOverall Confusion Matrix:")
print(overall_cm)

# Optional: Visualize the overall confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Overall Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(members))
plt.xticks(tick_marks, members, rotation=45)
plt.yticks(tick_marks, members)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()