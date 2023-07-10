import face_recognition
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN
import warnings
from tqdm import tqdm

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
