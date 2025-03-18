# this is for extracting the zip files in the dataset

import zipfile
import os

# Root directory of your dataset
root_dir = "NJZ_photo_dataset"

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png")

# Function to generate a unique filename if a duplicate exists
def get_unique_filename(dst_path):
    base, ext = os.path.splitext(dst_path)
    counter = 1
    new_path = dst_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

# Loop through each member's folder
for member_folder in os.listdir(root_dir):
    member_path = os.path.join(root_dir, member_folder)
    
    # Check if it's a directory (skip files in root)
    if os.path.isdir(member_path):
        print(f"Processing folder: {member_folder}")
        
        # Loop through files in the member's folder
        for filename in os.listdir(member_path):
            if filename.endswith(".zip"):  # Identify ZIP files
                zip_path = os.path.join(member_path, filename)
                print(f"  Unzipping {filename} in {member_folder}...")

                # Open and extract the ZIP file into the same folder
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract only image files
                    for file in zip_ref.namelist():
                        if file.lower().endswith(image_extensions):
                            # Initial destination path for the extracted file
                            dst_path = os.path.join(member_path, os.path.basename(file))
                            # If the file already exists, generate a unique name
                            if os.path.exists(dst_path):
                                unique_dst_path = get_unique_filename(dst_path)
                                print(f"  Renamed {file} to {os.path.basename(unique_dst_path)} (duplicate avoided)")
                                # Extract to a temporary location first
                                temp_extract_path = zip_ref.extract(file, member_path)
                                # Move to the unique destination
                                os.rename(temp_extract_path, unique_dst_path)
                            else:
                                zip_ref.extract(file, member_path)
                
                os.remove(zip_path)  # Deletes the ZIP file
                print(f"  Extracted {filename}")

        print(f"Finished processing {member_folder}\n")

print("All ZIP files processed!")

# --Photo number--

# Dictionary to store the count for each member
photo_counts = {}

# Loop through each member's folder
for member_folder in os.listdir(root_dir):
    member_path = os.path.join(root_dir, member_folder)
    
    # Check if it's a directory (skip files in root)
    if os.path.isdir(member_path):
        # Count images in the folder
        image_count = 0
        for filename in os.listdir(member_path):
            if filename.lower().endswith(image_extensions):
                image_count += 1
        photo_counts[member_folder] = image_count

# Display the results
print("Number of photos in each member's folder:")
for member, count in photo_counts.items():
    print(f"{member}: {count} photos")

# Calculate and display the total
total_photos = sum(photo_counts.values())
print(f"\nTotal number of photos: {total_photos}")