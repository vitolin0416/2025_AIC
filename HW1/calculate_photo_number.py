# this is for calculating the number of photos in each member's folder

import os

# Root directory of your dataset
# root_dir = "NJZ_photo_dataset"
root_dir = "NJZ_cropped_faces_dataset"

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png")

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
print(f"Number of photos in {root_dir} each member's folder:")
for member, count in photo_counts.items():
    print(f"{member}: {count} photos")

# Calculate and display the total
total_photos = sum(photo_counts.values())
print(f"\nTotal number of photos: {total_photos}")