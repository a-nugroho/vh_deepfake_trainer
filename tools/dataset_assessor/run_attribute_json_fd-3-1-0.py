import os
import sys
sys.path.append(r"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/utils")  # full path to the folder containing __init__.py
import json
import re
import cv2
import argparse
from tqdm import tqdm
from age_detection import AgeClassification
from face_quality_assessment import FaceQualityAssessment 
import csv
import stone
from utils import get_list_images
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import importlib.util
import importlib.machinery
from pathlib import Path
import copy

# map package name → alternative path
ALT_MODULE_PATHS = {
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection"
}

import sys

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

from face_detection import FaceDetection

# install our custom finder at the front
finder = AltImportFinder()
if not any(isinstance(f, AltImportFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, finder)

def read_data(root_json, name_directory):
    path_json = os.path.join(root_json,name_directory+".json")
    with open(path_json, 'r') as file:
        data_json = json.load(file)
    return data_json

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--json_path", required=True, help="Path to JSON data source")
    parser.add_argument("--dataset_name","-d", required=False, help="Dataset name")
    parser.add_argument("--dir_images", required=False, help="Folder for path images")
    args = parser.parse_args()
    return args

def label_decoding(label_name, label_type):
    if label_type.lower()=="pred_gender":
        if label_name==0:
            label_name = "FEMALE"
        elif label_name==1:
            label_name = "MALE"
        else:
            label_name = "OTHER"
        return label_name

def process_attribute(att_name,processor_att,data_input):
    if att_name in ["pred_age","pred_gender","pred_skin_tone"]:
        try: value_att = processor_att(data_input)
        except: value_att = "UNKNOWN"
    elif att_name in ["score_blur_face","score_dark","score_blur_img"]:
        try: value_att = processor_att(data_input)
        except: value_att = -100
    elif att_name in ["head_pose"]:
        try: value_att = processor_att(data_input)
        except: value_att = {"yaw":None, "pitch":None, "roll": None}
    return value_att

def func_tone(data_input):
    result = stone.process(data_input)
    value_att = result["faces"][0]["tone_label"]
    return value_att

def func_gender(data_input):
    faces = app.get(data_input)
    value_att = label_decoding(faces[0]['gender'],"pred_gender")
    return value_att

def func_headpose(data_input):
    image, dets = data_input
    outputs = model_fd.headpose_detector.infer(image, dets[:, :4])
    yaw, pitch, roll = outputs[0]
    return {"yaw":yaw.astype(float), "pitch":pitch.astype(float), "roll": roll.astype(float)}

def result_dumping(root_result, data_result):
    print(len(data_result))
    for att_now, dict_result in data_result.items():
        path_result_att = os.path.join(root_result,att_now+".json")
        os.makedirs(os.path.dirname(path_result_att), exist_ok=True)
        with open(path_result_att, "w") as f:
            json.dump(dict_result, f, indent=4)

def merge_nested_dicts(*dicts, mode="add_new"):
    """
    Merge multiple nested dictionaries without modifying the originals.

    Modes:
      - "add_new": add new top-level and nested keys
      - "existing_only": only merge into top-level keys that exist in the base dict,
                         but allow adding new nested keys inside those top-level keys.
    """
    if len(dicts) < 2:
        raise ValueError("Need at least two dictionaries to merge.")
    base = copy.deepcopy(dicts[0])

    def recursive_merge(target, source):
        """
        Add keys from source into target:
         - If both target[k] and source[k] are dicts -> recurse
         - If k not in target -> add deepcopy(source[k])
         - If k in target but not both dicts -> skip (no overwrite)
        """
        for k, v in source.items():
            if k in target:
                if isinstance(target[k], dict) and isinstance(v, dict):
                    recursive_merge(target[k], v)
                else:
                    # target has non-dict (or types mismatch) -> skip to avoid overwrite
                    continue
            else:
                # add new nested key
                target[k] = copy.deepcopy(v)

    for d in dicts[1:]:
        for top_k, top_v in d.items():
            if mode == "existing_only" and top_k not in base:
                # skip whole top-level key if it's not in base
                continue
            if top_k not in base and mode == "add_new":
                base[top_k] = copy.deepcopy(top_v)
            else:
                # top_k exists in base — merge nested keys if possible
                if isinstance(base[top_k], dict) and isinstance(top_v, dict):
                    recursive_merge(base[top_k], top_v)
                else:
                    # base[top_k] is not a dict -> cannot merge nested keys, skip
                    continue

    return base

def read_att_json(dir_att):
    files_json = [os.path.join(dir_att, f) for f in os.listdir(dir_att) if os.path.isfile(os.path.join(dir_att, f)) and f.lower().endswith(('.json'))]
    list_dict_att = []
    for f_n in files_json:
        att_name,_ = os.path.splitext(f_n)
        with open(f_n, "r") as f:
            dict_att = json.load(f)
        list_dict_att.append(dict_att)

    return merge_nested_dicts(*list_dict_att, mode="add_new")

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
        
    

"""
def custom_processor(processor_att,att_name):
    def func_gender(data_input):
        faces = processor_att(data_input)
        value_att = label_decoding(faces[0]['gender'],att_name)
        return value_att
    def func_tone(data_input):
        result = processor_att(data_input)
        value_att = result["faces"][0]["tone_label"]
        return value_att
    options = {
        "pred_gender": func_gender,
        "pred_skin_tone": func_tone,
    }

    return options.get(att_name.lower(), None)
"""
def main():
    # Init Models
    global app
    global model_fd

    model_fd = FaceDetection(use_cuda=True) # face detection wrapper
    model_ad = AgeClassification()
    model_qa = FaceQualityAssessment()
    ctx = 0
    app = FaceAnalysis(allowed_modules=['detection', 'genderage'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=ctx, det_size=(640,640))

    collection_processor = {"pred_skin_tone":func_tone,
                       "pred_gender":func_gender,
                       "head_pose":func_headpose,
                       "pred_age": model_ad.predict,
                       "score_blur_face":model_qa.blur.blur_score_selfie,
                       "score_dark":model_qa.dark.contrast_score_selfie,
                       "score_blur_img":model_qa.image_quality.quality_score_selfie}
    
    
    args = read_args()
    path_json_source = args.json_path
    # Access them
    print("JSON path:", path_json_source)
    if args.dataset_name:
        folder_name = args.dataset_name
    else:
        folder_name = path_json_source.split(".")[0].split("_")[-1]
    
    #root_result = f"dataset_assessor/results_att/{folder_name}/[attribute].json"
    #print(re.sub(r'\s+', '-', args.name_directory.split('/')[-1]))
    #path = f"results/{args.name_directory.split('/')[-2]}.json"

    #with open(path_json_source, "r") as f:
    #    data_source = json.load(f)
    #att_included = ["pred_skin_tone","pred_gender","head_pose"]
    att_included = ["pred_skin_tone","head_pose","pred_age","pred_gender"]
    dir_repo = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/"
    dir_att = os.path.join(dir_repo,f"dataset_assessor/results_att_fd3-1-0/{args.dataset_name}")
    dir_att_2 = os.path.join(dir_repo,f"dataset_assessor/results_att_fd3-1-0-temp/{args.dataset_name}")
    data_source = get_data_source(args.dataset_name,prefix="result_",
                    dir_att_list=[dir_att,dir_att_2])
    dict_extracted_ids = {k:[] for k in att_included}
    for att_now in dict_extracted_ids.keys():
        for k, v in data_source.items():
            if att_now in v:
                dict_extracted_ids[att_now].append(k)
    #if os.path.exists(path_result):
    #    with open(path_result, "r") as f:
    #        data_result = json.load(f)
    #else:
    #    data_result = {}
    list_path_images = list(data_source.keys())
    names_attribute = {j:k for j,k in collection_processor.items() if j in att_included}


    data_result = {}
    for att_now in att_included:
        data_result[att_now] = {}
    root_result = f"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0-24Nov/{folder_name}"
                
    # dict_result = one row of data
    for id_now,key_now in enumerate(tqdm(list_path_images)):
        if args.dir_images:
            #if defined, change the folder of all images to this one
            dir_name, file_name = os.path.split(key_now)
            key_now = os.path.join(args.dir_images, file_name)
        #if key_now in data_result:
        #    continue
        #else:
        dict_result_base = {"path":key_now}
        path_img = key_now
        
        # read sample image from file
        img = cv2.imread(path_img)

        if img is None:
            dict_result_base["status"] = "INVALID IMAGE PATH"
            
        try:    
            # get face detection result
            #dets, angle = model_fd.predict(img, strict_level="high")
            dets, angle = model_fd.predict(img)
            
            # Age
            #out = model_fd.align_single_face(img,dets,angle,outcolor="RGB",crop_size=300)
            #img_aligned = out[0]
            
            img_aligned = model_fd.extract_face(img,dets,angle, task="face-recognition")
            
            # get crop_single_face with loose_factor=1.25
            img_cropped = model_fd.extract_face(img, dets, angle, task="face-quality")
            

            dict_result_base["status"] = "FACE DETECTED"

            image, scale, pads = model_fd.processor.prepare_image(img)
        
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
                    #print(name_att)
                    if name_att in ["pred_age"]:
                        value_att = process_attribute(name_att,processor_att,img_aligned)
                    elif name_att in ["pred_gender"]:
                        value_att = process_attribute(name_att,processor_att,img)
                    elif name_att in ["score_blur_face","score_dark"]:
                        value_att = process_attribute(name_att,processor_att,img_cropped)
                    elif name_att in ["pred_skin_tone"]:
                        value_att = process_attribute(name_att,processor_att,path_img)
                    elif name_att in ["head_pose"]:
                        value_att = process_attribute(name_att,processor_att,[img,dets])

                    dict_result_att[name_att] = value_att
                    
            data_result[name_att][key_now] = dict_result_att

            
        # get the blur detection
        #is_blur_img = model_qa.image_quality.blur_detection(blur_score_img)

        # get deepfake decision
        #is_deepfake = model_fdd.classify_predictions(deepfake_score)

        #print(f"{is_deepfake}")
        # Saving result
        # save to json
        if id_now%50 == 0:
            result_dumping(root_result, data_result)
            #for att_now in att_included:
                
if __name__ == "__main__":
    main()

        