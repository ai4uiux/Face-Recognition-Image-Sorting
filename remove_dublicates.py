"""
Summary:

1. Project Exploration: The initial project (https://github.com/ageitgey/face_recognition/tree/master) is a library for face recognition in Python. It uses dlib's state-of-the-art face recognition built with deep learning to provide high accuracy in face detection and recognition.

2. Initial Script Creation: Based on the understanding of the project, an initial Python script was created that uses the face_recognition library to detect faces in images and group them based on similarity. The script was designed to ask for an input directory (containing the images) and an output directory (where the sorted images would be stored) through a popup dialog.

3. Script Testing and Debugging: The script was tested on a local machine and encountered an error related to a circular import in the face_recognition library. The error was due to the script file being named `face_recognition.py`, which conflicted with the face_recognition library. The script was renamed to `face_sorter.py` to resolve this issue.

4. Script Enhancement: The script was enhanced to create fewer folders by grouping similar-looking faces into the same folder. To achieve this, the DBSCAN clustering algorithm from the scikit-learn library was introduced to cluster similar face encodings together.

5. Dependency Resolution: An error related to a missing DLL file required by the scikit-learn library was encountered. The issue was resolved by installing the latest Microsoft Visual C++ Redistributable for Visual Studio.

6. Script Refinement: The script was refined to be able to select images from subdirectories in the input directory. The script was modified to walk through the directory tree of the input directory and process all images found.

7. Progress Tracking: A progress bar was added to display the progress of the script in percentage, and a completion message was added at the end of the script.

8. Duplicate Image Removal: The script was enhanced to find and remove duplicate images. The imagehash library was introduced to calculate a hash for each image, and duplicate images (with the same hash) were skipped.

9. Final Script: The final script sorts images based on the faces they contain, removes duplicate images, and provides progress tracking. It uses the face_recognition library for face detection and encoding, the scikit-learn library for clustering face encodings, the tqdm library for progress tracking, and the imagehash library for duplicate image detection.
"""

import face_recognition
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN
import warnings
from tqdm import tqdm
from PIL import Image
import imagehash

# Suppress warnings from joblib
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

# Ask the user for the input directory
print("Please select the input directory.")
image_dir = filedialog.askdirectory(title="Select Input Directory")

# Ask the user for the output directory
print("Please select the output directory.")
output_dir = filedialog.askdirectory(title="Select Output Directory")

# Initialize an empty list for face encodings and labels
encodings = []
labels = []
hashes = []

# Get the total number of images
total_images = sum([len(files) for r, d, files in os.walk(image_dir)])

# Initialize the progress bar
pbar = tqdm(total=total_images, desc="Processing images")

# Loop over each file in the directory tree
for dirpath, dirnames, filenames in os.walk(image_dir):
    for filename in filenames:
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(dirpath, filename)
            image = face_recognition.load_image_file(image_path)

            # Calculate the hash of the image
            image_hash = imagehash.average_hash(Image.open(image_path))

            # If this hash is in our list of hashes, it's a duplicate
            if image_hash in hashes:
                pbar.update(1)
                continue

            # Add the hash to our list of hashes
            hashes.append(image_hash)

            # Detect face in the image and get its encoding
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # If no face is found, skip this image
            if len(face_encodings) == 0:
                pbar.update(1)
                continue

            # For this example, we are only considering the first face found in the image
            face_encoding = face_encodings[0]

            # Add the face encoding and label to our lists
            encodings.append(face_encoding)
            labels.append(image_path)

            # Update the progress bar
            pbar.update(1)

# Close the progress bar
pbar.close()

# Use DBSCAN to cluster the face encodings
print("Clustering faces...")
clt = DBSCAN(metric="euclidean")
clt.fit(encodings)

# Loop over each detected face
for label, enc in zip(labels, clt.labels_):
    # Create a directory for this face, if it doesn't exist
    label_dir = os.path.join(output_dir, f'person_{enc}')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Copy the image to the appropriate directory
    shutil.copyfile(label, os.path.join(label_dir, os.path.basename(label)))

print("Sorting completed!")
