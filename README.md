# Face Recognition Image Sorting

This repository contains a Python script that uses artificial intelligence to identify individuals in a large number of images and sort them into respective folders.

## How it Works

The script uses the face_recognition library to detect faces in images and group them based on similarity. It also uses the DBSCAN clustering algorithm from the scikit-learn library to group similar-looking faces into the same folder. The script is user-friendly, asking for the input and output directories through a popup dialog. It processes all images in the input directory and its subdirectories, and copies the sorted images to the output directory, grouped into subdirectories based on the face they contain.

## Installation

1. Clone this repository to your local machine.
2. Install the required Python libraries by running `pip install -r requirements.txt` in your terminal.

## Usage

1. Run the script by typing `python face_sorter.py` in your terminal.
2. When prompted, select the input directory (containing the images) and the output directory (where the sorted images will be stored).

## Dependencies

- face_recognition
- scikit-learn
- tkinter
- tqdm
- PIL
- imagehash

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.
