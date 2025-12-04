import os
import logging
import json
import cv2

from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import sys
import importlib.util
import importlib.machinery


# map package name → alternative path
ALT_MODULE_PATHS = {
    #"face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-0-0/face-production-face-detection/face_detection"
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-0-0/face-production-face-detection/face_detection"
}

class AltImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ALT_MODULE_PATHS:
            alt_path = Path(ALT_MODULE_PATHS[fullname])
            init_file = alt_path / "__init__.py"
            if not init_file.exists():
                return None
            loader = importlib.machinery.SourceFileLoader(fullname, str(init_file))
            return importlib.util.spec_from_loader(fullname, loader, origin=str(init_file))
        return None

# install our custom finder at the front
finder = AltImportFinder()
if not any(isinstance(f, AltImportFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, finder)

from face_detection import FaceDetection
# Initialize face detection (CPU or GPU)
fd = FaceDetection(use_cuda=True)

path_folder_target = "train"

dataset_info = {}
index = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--base_dir", type=str)
    parser.add_argument("-o","--output_json",required=True, type=str)
    args = parser.parse_args()
    return args

args = get_args()
output_json = args.output_json

# Setup logging to a file
logging.basicConfig(
    filename=f'file_read_log_{Path(output_json).stem}.txt',
    filemode='w',  # Overwrite each time; use 'a' to append
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Function to extract label and method from the path
def get_label_and_method(image_path):
    """Determines label (real=0, fake=1) and deepfake method from path."""
    path_parts = image_path.split(os.sep)  # Split path into components
    # Default values
    #method = "ReSwapper, refined blending mask, CodeFormer"
    label = None
    ## Check if "fake" exists in path
    if "fake" in path_parts:
        label = 1  # Fake image
        
        # Extract deepfake method (assumed to be the directory after "fake")
        fake_index = path_parts.index("fake")
        if fake_index + 1 < len(path_parts):  # Ensure there's another directory after "fake"
            method = path_parts[fake_index+1]
        else:
            method = "unknown"  # Take the method name
    else:
        label = 0
        method = "in_the_wild_live"
    #label = 1

    return label, method


base_dir = args.base_dir
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
for root, _, files in os.walk(base_dir):
    print(root)
    for file in tqdm(files):
        if file.lower().endswith(valid_exts):  # Process only images
            image_path = os.path.join(root, file)
            relative_path = os.path.join(f"{base_dir}", os.path.relpath(image_path, base_dir))
            #print(relative_path)
            try:
                # Read the image
                img = cv2.imread(image_path)

                # Check if the image was successfully loaded
                if img is None:
                    continue  # Skip if image couldn't be loaded

                # Use FD 3.0.0
                # Perform face detection
                dets, angle = fd.predict(img, strict_level="low")
                img_cropped = fd.extract_face(img, dets, angle, task="face-iso", return_bbox=True, loose_factor = 0.8)
                #img_cropped = fd.extract_face(img, dets, angle, task="face-iso", return_bbox=True, loose_factor = 1.5)
    
                if len(dets) <= 0:
                    continue  # Skip if no face detected

                # Convert NumPy arrays to lists
                dets_list = dets.tolist() if isinstance(dets, np.ndarray) else dets
                angle_list = angle.tolist() if isinstance(angle, np.ndarray) else angle

                # Get label and method
                label, method = get_label_and_method(relative_path)
                #label=1
                #method="ReSwapper Codeformer Enhanced"

                #if label == 0:
                #    method = "in_the_wild_live"

                # Save the extracted data
                dataset_info[index] = {
                    "image_path": relative_path,
                    "label": label,
                    "method": method,
                    "dets": dets_list,
                    "angle": angle_list
                }

                # Log file successfully read
                logging.info(f"Processed file: {relative_path} successfully")

                index += 1  # Update index
            except Exception as e:
                # Log any error that occurs
                logging.error(f"Failed to Process file: {relative_path} | Error: {e}")
                

print(f"Total processed images: {index}")

# Save to JSON file
with open(output_json, "w") as f:
    json.dump(dataset_info, f, indent=4)

print(f"Processing complete. Data saved to {output_json}")