import os
import sys
sys.path.insert(0, '/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools') 
import importlib.util
import importlib.machinery
from pathlib import Path
import argparse
from utils.json_utils import read_att_json, merge_nested_dicts
import json
import cv2
# map package name → alternative path
ALT_MODULE_PATHS = {
    "face_deepfake": "/mnt/ssd/workspace/adi/vh_repos_byversion/deepfake-detection/1-1-0/face-production-face-deepfake/face_deepfake",
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection"
}

import sys

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

from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from face_detection import FaceDetection
from skimage import transform as trans
from face_deepfake import DeepfakeDetection
import dlib
from datetime import datetime

def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        landmark = predictor(cropped_face, face_align[0])
        landmark = shape_to_np(landmark)

        return cropped_face, landmark,face
    
    else:
        return None, None

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def resize_and_center_content_opencv(img, target_size=224, threshold=5):
    """
    Resize an image while maintaining aspect ratio, detect content, 
    center it on a black square canvas of size (target_size x target_size).
    
    Parameters:
        img (np.ndarray): Input BGR image (OpenCV format).
        target_size (int): Desired output size (square).
        threshold (int): Intensity threshold for content detection.
    
    Returns:
        np.ndarray: Processed image (target_size x target_size x 3).
    """
    # Step 1: Convert to grayscale for content detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Find non-black area using threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)

    if coords is None:
        # Image is all black, return centered black image
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    # Step 3: Resize while maintaining aspect ratio
    h_c, w_c = cropped.shape[:2]
    scale = target_size / max(h_c, w_c)
    resized = cv2.resize(cropped, (int(w_c * scale), int(h_c * scale)), interpolation=cv2.INTER_CUBIC)

    # Step 4: Create black canvas and center the resized image
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    h_r, w_r = resized.shape[:2]
    top = (target_size - h_r) // 2
    left = (target_size - w_r) // 2

    canvas[top:top+h_r, left:left+w_r] = resized
    return canvas

def result_dumping(root_result, data_result):
    for att_now, dict_result in data_result.items():
        path_result_att = os.path.join(root_result,att_now+".json")
        os.makedirs(os.path.dirname(path_result_att), exist_ok=True)
        with open(path_result_att, "w") as f:
            json.dump(dict_result, f, indent=4)

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--json_path", required=True, help="Path to JSON data source")
    parser.add_argument("--dataset_name","-d", required=False, help="Dataset name")
    parser.add_argument("--dir_images", required=False, help="Folder for path images")
    args = parser.parse_args()
    return args

def get_data_source(dataset_name,dir_att_list=None,prefix=None):
    if prefix:
        json_file_name = str(prefix)+dataset_name
    else:
        json_file_name = dataset_name
    path_json = os.path.join("/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor",f"{json_file_name}.json")
    
    with open(path_json, 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_source = json.load(f)

    if "200k_live_face_dataset" in path_json:
        data_source_new = {}
        for k, v in data_source.items():
            k_new = k.replace("/processes","")
            data_source_new[k_new] = v
        
        data_source = data_source_new

    if dir_att_list:
        if isinstance(dir_att_list, str):
            dir_att_list = [dir_att_list]

        list_dict_att_json = []
        for dir_att in dir_att_list:
            list_dict_att_json.append(read_att_json(dir_att))

        data_source = merge_nested_dicts(*[data_source]+list_dict_att_json,mode="existing_only")
    
    return data_source

def process_attribute(att_name,processor_att,data_input):
    if att_name in ["pred_age","pred_gender","pred_skin_tone"]:
        try: value_att = processor_att(data_input)
        except: value_att = "UNKNOWN"
    elif att_name in ["dfd_1-0-0"]:
        try: value_att = float(processor_att(data_input))
        except: value_att = -100
    elif att_name in ["head_pose"]:
        try: value_att = processor_att(data_input)
        except: value_att = {"yaw":None, "pitch":None, "roll": None}
    return value_att

def main():
    global model_fd
    global model_dfd

    model_fd = FaceDetection(use_cuda=True) # face detection wrapper    
    model_dfd = DeepfakeDetection(device_name="cuda")
    collection_processor = {"dfd_1-0-0":model_dfd.predict}
    args = read_args()
    path_json_source = args.json_path
    # Access them
    print("JSON path:", path_json_source)
    if args.dataset_name:
        folder_name = args.dataset_name
    else:
        folder_name = path_json_source.split(".")[0].split("_")[-1]


    timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    # Save all prints to output.txt
    sys.stdout = Logger(f"logs/{Path(__file__).stem}o.txt")

    # Sanity check for DFD version
    print(f"Engine 1: {str(model_dfd.engine_1)}")
    print(f"Engine 2: {str(model_dfd.engine_2)}")

    num_pred_deepfake = 0
    num_pred_real = 0
    num_real = 0
    num_deepfake = 0

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

    import os
    print(os.getcwd())


    dir_repo = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/"
    att_included = ["dfd_1-0-0"]
    dir_att = os.path.join(dir_repo,f"dataset_assessor/results_att_fd3-1-0/{args.dataset_name}")
    path_json = args.json_path
    with open(path_json, 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_source = json.load(f)
    dict_extracted_ids = {k:[] for k in att_included}
    for att_now in dict_extracted_ids.keys():
        for k, v in data_source.items():
            if att_now in v:
                dict_extracted_ids[att_now].append(k)
    
    list_path_images = list(data_source.keys())
    names_attribute = {j:k for j,k in collection_processor.items() if j in att_included}
    data_result = {}
    for att_now in att_included:
        data_result[att_now] = {}
    
    root_result = f"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/{folder_name}"
    

    for id_now,key_now in enumerate(tqdm(list_path_images)):
        if args.dir_images:
            #if defined, change the folder of all images to this one
            dir_name, file_name = os.path.split(key_now)
            key_now = os.path.join(args.dir_images, file_name)

        #print(key_now)
        #print(data_source[key_now])
        dict_result_base = {"path":key_now}
        #path_img = key_now
        path_img = data_source[key_now]["processed_path"]
        dict_result_base["processed_path"] = path_img
        
        
        # read sample image from file
        img = cv2.imread(path_img)

        if img is None:
            dict_result_base["status"] = "INVALID IMAGE PATH"

        try:    
            # get face detection result
            #dets, angle = model_fd.predict(img, strict_level="high")
            # Processed path no need to crop
            #dets, angle = model_fd.predict(img)
            #img_cropped = model_fd.extract_face(img, dets, angle, task="face-deepfake")
            
            img_cropped = img
            dict_result_base["status"] = "FACE DETECTED"

        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
            raise  # re-raise if you want the program to exit immediately 

        except:
            dict_result_base["status"] = "FACE NOT DETECTED"
        
        for name_att, processor_att in names_attribute.items():
            dict_result_att = dict_result_base.copy()
            
            if dict_result_base["status"] == "FACE DETECTED":
                if key_now in dict_extracted_ids[name_att]:
                    dict_result_att[name_att] = data_source[key_now][name_att]
                
                else:
                    if name_att in ["dfd_1-0-0"]:
                        value_att = process_attribute(name_att,processor_att,img_cropped)

                    dict_result_att[name_att] = value_att

            data_result[name_att][key_now] = dict_result_att

        if (id_now+1)%50 == 0:
            result_dumping(root_result, data_result)
            #for att_now in att_included:

    result_dumping(root_result, data_result)
    
if __name__ == "__main__":
    main()