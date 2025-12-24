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
from datetime import datetime
"""
1. Read JSON source OR folder dir
2. Perform alignment and cropping
3. Save to a folder
4. Create mew dataset's JSON

"""

# map package name → alternative path
ALT_MODULE_PATHS = {
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection"
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

class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.log = open(logfile, "a", encoding="utf-8")

    def _timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_terminal(self, msg):
        sys.stdout.write(msg)
        sys.stdout.flush()

    def _write_file(self, msg):
        self.log.write(msg)
        self.log.flush()   # <-- ensure immediate write to file

    def log_msg(self, message, mode="both"):
        """
        mode = "terminal", "file", "both"
        """
        if not message.endswith("\n"):
            message += "\n"

        msg = f"[{self._timestamp()}] {message}"

        if mode in ("terminal", "both"):
            self._write_terminal(msg)

        if mode in ("file", "both"):
            self._write_file(msg)

    # --- Shortcuts ---
    def t(self, message):      # terminal only
        self.log_msg(message, mode="terminal")

    def f(self, message):      # file only
        self.log_msg(message, mode="file")

    def b(self, message):      # both
        self.log_msg(message, mode="both")

    def close(self):
        self.log.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--base_dir", type=str)
    parser.add_argument("-o","--output_json",required=True, type=str)
    parser.add_argument("-t","--target_dir", type=str)
    args = parser.parse_args()
    return args

# Function to extract label and method from the path

def get_source_paths(base_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    file_list = []

    #for (root,dirs,files) in os.walk(base_dir):
    #    for file in files:
    #        file_list.append(file)
    #    root = root
    file_list = os.listdir(base_dir)         
    file_list = [f for f in file_list if f.lower().endswith(valid_exts)]

    return base_dir, file_list

def get_label_and_method(image_path):
    """Determines label (real=0, fake=1) and deepfake method from path."""
    path_parts = image_path.split(os.sep)  # Split path into components
    # Default values
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

def read_align_image(fd, image_path, align_face=False):
    # Read the image
    img = cv2.imread(image_path)
    # Check if the image was successfully loaded
    if img is None:
        return None, None, None, None
    
    # Use FD 3.0.0
    # Perform face detection
    dets, angle = fd.predict(img, strict_level="low")
    if len(dets) <= 0:
        return None, None, None, None
    
    #img_cropped = fd.extract_face(img, dets, angle, task="face-iso", return_bbox=True, loose_factor = 0.8,square_crop=True)
    #img_cropped = fd.extract_face(img, dets, angle, task="face-iso", loose_factor = 0.8)
    img_cropped = fd.extract_face(img, dets, angle, task="face-deepfake")
    #img_cropped = fd.extract_face(img, dets, angle, task="face-iso", return_bbox=True, loose_factor = 0.8)
        
    #img_cropped = fd.extract_face(img, dets, angle, task="face-iso", return_bbox=True, loose_factor = 1.5)
    if not align_face:
        img_cropped = img
    # Convert NumPy arrays to lists
    
    return img_cropped, dets, angle, img

def save_image(output_dir, image_name, img_cropped):
    target_path = os.path.join(output_dir,image_name)
    cv2.imwrite(target_path,img_cropped)
    return target_path

if __name__ == "__main__":
    # Initialize face detection (CPU or GPU)
    fd = FaceDetection(use_cuda=True)

    path_folder_target = "train"

    dataset_info = {}
    index = 0

    args = get_args()
    output_json = args.output_json
    os.makedirs(args.target_dir,exist_ok=True)
    

    filename = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    logger = Logger(f"{filename}__{timestamp}.txt")

    base_dir = args.base_dir
    root, list_files = get_source_paths(args.base_dir)
    
    for file in tqdm(list_files):
        image_path = os.path.join(root, file)
        relative_path = os.path.join(f"{base_dir}", os.path.relpath(image_path, base_dir))
        #print(relative_path)
        #try:
        
        if "uniface" in base_dir or "blendface" in base_dir:
            align_face=False
        else:
            align_face=True
            
        img_cropped, dets, angle, image = read_align_image(fd,image_path,align_face)
        yaw, pitch, roll = None, None, None

        
        if dets is not None:
            outputs = fd.headpose_detector.infer(image, dets[:, :4])
            yaw, pitch, roll = outputs[0]
            if abs(yaw)>10 or abs(pitch)>10 or abs(roll)>10:
                continue

        else:
            continue

        # Get label and method
        label, method = get_label_and_method(relative_path)
        
        target_path = save_image(args.target_dir, file, img_cropped)
        dets = dets.tolist() if isinstance(dets, np.ndarray) else dets
        angle = angle.tolist() if isinstance(angle, np.ndarray) else angle

        # Save the extracted data
        dataset_info[index] = {
            "image_path": relative_path,
            "processed_path": target_path,
            "label": label,
            "method": method,
            "dets": dets,
            "angle": angle,
            "head_pose":{"yaw":yaw.astype(float) if yaw else None, 
                            "pitch":pitch.astype(float) if pitch else None, 
                            "roll": roll.astype(float) if roll else None}
        }
        
        # Log file successfully read
        #logging.info(f"Processed file: {relative_path} successfully")
        logger.f(f"Processed file: {relative_path} successfully")

        index += 1  # Update index
        #except Exception as e:
        #    # Log any error that occurs
        #    logging.error(f"Failed to Process file: {relative_path} | Error: {e}")
            

    print(f"Total processed images: {index}")

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(dataset_info, f, indent=4)

    print(f"Processing complete. Data saved to {output_json}")