# this is for the face detection and cropping

import os
import cv2
from mtcnn import MTCNN
import numpy as np
import shutil

# Root directories
input_root_dir = "Hyein_renamed_photo_dataset"  # Renamed dataset with ASCII names
output_root_dir = "NJZ_cropped_faces_dataset"  # Cropped faces
original_successful_dir = "NJZ_original_successful_dataset"  # Originals of successful crops

# List of members to process (modify this as needed)
members_to_process = ["Hyein"]  # Example: process Minji and Hanni only

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png")

# Initialize MTCNN face detector
detector = MTCNN()

# Create output directories
if not os.path.exists(output_root_dir):
    os.makedirs(output_root_dir)
if not os.path.exists(original_successful_dir):
    os.makedirs(original_successful_dir)

# Function to generate a unique filename
def get_unique_filename(base_path, filename):
    base, ext = os.path.splitext(filename)  # Split into name and extension (e.g., "hanni_001", ".jpg")
    # Extract the original number (e.g., "001" from "hanni_001")
    prefix, num = base.rsplit("_", 1) if "_" in base and base.rsplit("_", 1)[-1].isdigit() else (base, "0")
    counter = 1
    new_filename = filename
    new_path = os.path.join(base_path, new_filename)
    
    while os.path.exists(new_path):
        new_filename = f"{prefix}_{counter}_{num}{ext}"  # e.g., "hanni_1_001.jpg"
        new_path = os.path.join(base_path, new_filename)
        counter += 1
    
    return new_filename

# Function to crop face from an image
def crop_face(image_path, output_path, original_output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return False
    
    # Convert to RGB (MTCNN expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return False
    
    # Use the first detected face (assume one face per image)
    face = faces[0]
    x, y, w, h = face['box']
    
    # Add padding around the face (20% of the bounding box size)
    padding = int(max(w, h) * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Crop the face
    cropped = img[y:y+h, x:x+w]
    
    # Save the cropped image with unique name
    unique_output_filename = get_unique_filename(os.path.dirname(output_path), os.path.basename(output_path))
    unique_output_path = os.path.join(os.path.dirname(output_path), unique_output_filename)
    cv2.imwrite(unique_output_path, cropped)
    
    # Copy the original image with unique name
    unique_original_filename = get_unique_filename(os.path.dirname(original_output_path), os.path.basename(original_output_path))
    unique_original_output_path = os.path.join(os.path.dirname(original_output_path), unique_original_filename)
    shutil.copy2(image_path, unique_original_output_path)
    
    if unique_output_filename != os.path.basename(output_path):
        print(f"  Renamed {os.path.basename(output_path)} to {unique_output_filename} in cropped folder")
    if unique_original_filename != os.path.basename(original_output_path):
        print(f"  Renamed {os.path.basename(original_output_path)} to {unique_original_filename} in original folder")
    
    return True

# Process specified members
total_successful_crops = 0
total_failed_crops = 0

for member_to_process in members_to_process:
    member_input_path = os.path.join(input_root_dir, member_to_process)
    member_output_path = os.path.join(output_root_dir, member_to_process)
    original_member_output_path = os.path.join(original_successful_dir, member_to_process)

    if os.path.isdir(member_input_path):
        # Create output folders
        if not os.path.exists(member_output_path):
            os.makedirs(member_output_path)
        if not os.path.exists(original_member_output_path):
            os.makedirs(original_member_output_path)
        
        print(f"Processing {member_to_process}...")
        total_images = 0
        successful_crops = 0
        failed_crops = 0
        
        # Loop through images
        for filename in os.listdir(member_input_path):
            if filename.lower().endswith(image_extensions):
                total_images += 1
                input_path = os.path.join(member_input_path, filename)
                output_path = os.path.join(member_output_path, filename)
                original_output_path = os.path.join(original_member_output_path, filename)
                
                # Crop and save the face
                if crop_face(input_path, output_path, original_output_path):
                    successful_crops += 1
                else:
                    failed_crops += 1
        
        # Update global counts
        total_successful_crops += successful_crops
        total_failed_crops += failed_crops
        
        # Print results for this member
        print(f"  {member_to_process}: {successful_crops}/{total_images} images cropped successfully")
        print(f"  Successful crops: {successful_crops}")
        print(f"  Failed crops: {failed_crops}\n")
    else:
        print(f"Folder {member_to_process} not found in {input_root_dir}")

# Print overall results
print(f"Overall results:")
print(f"  Total successful crops: {total_successful_crops}")
print(f"  Total failed crops: {total_failed_crops}")
print("Face cropping completed!")