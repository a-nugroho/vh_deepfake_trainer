import argparse
import sys
import importlib.util
import importlib.machinery
from importlib import abc 
from pathlib import Path
import glob
import os
import cv2
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
# map package name → alternative path
ALT_MODULE_PATHS = {
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection",
    "face_deepfake": "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/model_production/v-1-2-0/face-production-face-deepfake/face_deepfake",
    "face_deepfake_old": "/mnt/ssd/workspace/adi/vh_repos_byversion/deepfake-detection/1-0-0/face-production-face-deepfake/face_deepfake"
}

import base64

class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()   # <-- force immediate write
        
    def flush(self):
        # required for Python's internal flush calls
        self.terminal.flush()
        self.log.flush()

class AltImportFinder(abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ALT_MODULE_PATHS:
            alt_path = Path(ALT_MODULE_PATHS[fullname])
            init_file = alt_path / "__init__.py"
            print(alt_path)
            if not init_file.exists():
                return None
            loader = importlib.machinery.SourceFileLoader(fullname, str(init_file))
            return importlib.util.spec_from_loader(fullname, loader, origin=str(init_file))
        return None


def get_all_images_glob(directory_path, extensions=['.png', '.jpg', '.jpeg','.txt']):
    """
    Recursively finds all files with specified image extensions using glob.
    """
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory_path, '**', f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=True))
    return image_files

def read_base64_image_from_txt(filepath):
    """
    Reads a base64 encoded image string from a text file, decodes it, 
    and converts it into an OpenCV image (numpy array).
    """
    try:
        with open(filepath, 'r') as f:
            base64_string = f.read().strip()

        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]

        base64_string = f"{base64_string}{'=' * (len(base64_string) % 4)}"
        
        # Decode the base64 string into bytes
        decoded_bytes = base64.b64decode(base64_string)

        # Convert the bytes to a 1D numpy array
        np_array = np.frombuffer(decoded_bytes, dtype=np.uint8)

        # Decode the numpy array as an image using OpenCV
        image = cv2.imdecode(np_array, flags=cv2.IMREAD_COLOR)

        if image is None:
            print("Error: cv2.imdecode() failed. Check if the base64 string is valid and correctly formatted.")
        else:
            return image

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

class JSONDataset(Dataset):
    def __init__(self, json_file, is_cropped=False, processed_path_key = "processed_path"):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.image_paths = list(self.data.keys())
        #self.labels = [self.data[path]["label"] for path in self.image_paths]
        self.processed_paths = [
            self.data[path][processed_path_key] for path in self.image_paths
        ]

        self.is_cropped = is_cropped

    def __len__(self):
        return len(self.image_paths)

    def get_preprocessed_path(self, image_path):
        """Generate a correct path for preprocessed images."""
        dataset_root = "/mnt/SSD/dataset/deepfake"

        if "UADFV" in image_path and "real" in image_path:
            image_path = image_path.replace("face/", "")
            return os.path.join(dataset_root, image_path)

        if os.path.isabs(image_path):
            return image_path

        return os.path.join(dataset_root, image_path)

    def __getitem__(self, idx):
        image_path = self.processed_paths[idx]
        label = self.labels[idx]

        image_path = self.get_preprocessed_path(image_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image not found: {image_path}")
            return self.__getitem__(idx + 1)

        if not self.is_cropped:
            dets, ang = fd.predict(image)
            if len(dets) == 0:
                print(f"No face detected in {image_path}")
                return self.__getitem__(idx + 1)
            cropped_image, bbox = fd.crop_single_face_square(
                image, dets, ang, loose_factor=1.3, crop_size=None
            )
            image = cropped_image

        return image, label

# install our custom finder at the front
finder = AltImportFinder()
#if not any(isinstance(f, AltImportFinder) for f in sys.meta_path):
sys.meta_path.insert(0, finder)

from datetime import datetime

import face_detection
from face_detection import FaceDetection
print(face_detection.__version__)
from face_deepfake.main import softmax
from face_deepfake import DeepfakeDetection

timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
# Save all prints to output.txt
sys.stdout = Logger(f"logs/{Path(__file__).stem}.txt")
fd = FaceDetection(use_cuda=True)
#engine_props = [{'input_size':(224, 224),'input_mean':[0.5, 0.5, 0.5],'input_std':[0.5, 0.5, 0.5],'model_path':'clip_df40_large.onnx'}, # effort
#{'input_size':(384, 384),'input_mean':[0.485, 0.456, 0.406],'input_std':[0.229, 0.224, 0.225],'model_path':'convnext_xlarge_384_in22ft1k_30.pth.onnx'},
##{'input_size':(224, 224),'input_mean':[0.5, 0.5, 0.5],'input_std':[0.5, 0.5, 0.5],'model_path':'clip_detector_optimized.onnx'},
#{'input_size':(224, 224),'input_mean':[0.48145466, 0.4578275, 0.40821073],'input_std':[0.26862954, 0.26130258, 0.27577711],'model_path':'effort_vh_1218.onnx'} # effort-VH
#]
list_models = ["clip_det","df40_pre","vh_effort","ensembled"]
fdd = DeepfakeDetection(device_name="cuda")
#fdd_100=DeepfakeDetection100(device_name="cuda")

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--source_path")
    parser.add_argument("--is_cropped", action="store_true", help="Use cropped images")
    #parser.add_argument("--json_result", "-r", required=True, help="JSON Result name")
    args = parser.parse_args()
    return args

args = read_args()
num_pred_deepfake = 0
num_pred_real = 0
num_real = 0
num_deepfake = 0

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp','.txt')
source_path = args.source_path
#processed_path_key = "processed"
for processed_path_key in ["processed","iso_aligned"]:

    if os.path.isdir(source_path):
        img_list = get_all_images_glob(source_path, valid_exts)
    elif os.path.isfile(source_path):
        if source_path.lower().endswith('.json'):
            dataset_now = JSONDataset(source_path,is_cropped=args.is_cropped,processed_path_key=processed_path_key)
            img_list = dataset_now.processed_paths
    #if args.folder_paths

    file_output = "_".join([Path(__file__).stem,"output",processed_path_key,Path(source_path.lower().split('/')[-1]).stem+".txt"])
    with open(file_output, "w", encoding="utf-8") as f:
        line = '|'.join(["image_name","pred_label","deepfake_score"])
        f.write(line + "\n")  # write to file

    #for folder_path in folder_paths:

    for img_file in img_list:
        #try:
        #img_path = os.path.join(os.getcwd(),folder_path,img_file)
        img_path = img_file
        #try:

        # read sample image from file
        if img_path.endswith(".txt"):
            img = read_base64_image_from_txt(img_path)
        else:
            img = cv2.imread(img_path)

        if(img is None):
            continue

        if args.is_cropped:
            img_cropped = img
        else:
            # get face detection result
            try:
                #dets, ang = fd.predict(img, strict_level="low") # for indonesian deepfake dataset v2
                dets, ang = fd.predict(img, strict_level="medium")
            except:
                continue

            if len(dets)==0:
                continue

            img_cropped = fd.extract_face(img, dets, ang, task="face-deepfake")
            #cv2.imwrite(f"pics/{img_path.split('/')[-1]}",img_cropped)
            
            #img_cropped = fd.extract_face(img, dets, ang, task="face-iso")
            #img_cropped = resize_and_center_content_opencv(img_cropped)
            
        # get deepfake score
        deepfake_score, score_all = fdd.predict(img_cropped, return_all= True)
        #deepfake_100_score = fdd_100.predict(img_cropped)

        #logits_1 = fdd.engine_1.predict(img_cropped)[0]
        #logits_2 = fdd.engine_2.predict(img_cropped)[0]
        #logits_3 = fdd.engine_3.predict(img_cropped)[0]
        
        # get deepfake decision
        is_deepfake = fdd.classify_predictions(deepfake_score)
        #is_deepfake_100 = fdd_100.classify_predictions(deepfake_100_score)
        #print(f"Sample {img_file} 1.0.0 = {'DEEPFAKE' if is_deepfake_100 else 'REAL'}, SCORE: {deepfake_100_score:.4f} 1.2.0 = {'DEEPFAKE' if is_deepfake else 'REAL'}, SCORE: {deepfake_score:.4f}")
        print(f"Sample {img_file} 1.2.0 = {'DEEPFAKE' if is_deepfake else 'REAL'}, SCORE: {deepfake_score:.4f}")
        #print(f"Logits 1: {logits_1}, Logits 2: {logits_2}, Logits 3: {logits_3}")
        #except:
        #    print(f"Cant process path {img_path}")
        num_pred_deepfake += int(is_deepfake)
        num_pred_real += int(not is_deepfake)

        #line = f"{img_file}|{'DEEPFAKE' if is_deepfake else 'REAL'}|{deepfake_score}"
        line = '|'.join([img_file,'DEEPFAKE' if is_deepfake else 'REAL',str(np.round(deepfake_score,5))])
        #line = '|'.join([img_file,'DEEPFAKE' if is_deepfake else 'REAL',str(np.round(deepfake_100_score,5)),str(np.round(deepfake_score,5))]+
        #                [str(np.round(softmax(i)[1].item(),5)) for i in score_all]+
        #                [str(np.round(i[0],5))+','+str(np.round(i[1],5)) for i in score_all])
        with open(file_output, "a") as f:
            f.write(f"{line}\n")

        #except:
        #    continue

    print(f"Pred Deepfake: {num_pred_deepfake} {100*num_pred_deepfake/(num_pred_deepfake+num_pred_real)}%")
    print(f"Pred Real: {num_pred_real} {100*num_pred_real/(num_pred_deepfake+num_pred_real)}%")