import os
import sys
sys.path.append(r"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/utils")  # full path to the folder containing __init__.py
import json
import re
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import csv
from utils import get_list_images
import importlib.util
import importlib.machinery
from importlib import abc  
from pathlib import Path
root_json = "/mnt/ssd/datasets/deepfake/dataset_json"

ALT_MODULE_PATHS = {
    "face_detection": "/mnt/ssd/workspace/adi/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection"
}
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

# install our custom finder at the front
finder = AltImportFinder()
#if not any(isinstance(f, AltImportFinder) for f in sys.meta_path):
sys.meta_path.insert(0, finder)

from face_detection import FaceDetection

def read_data(root_json, name_directory):
    path_json = os.path.join(root_json,name_directory+".json")
    with open(path_json, 'r') as file:
        data_json = json.load(file)
    return data_json

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--source_directory", "-d", required=True, help="Dataset folder")
    parser.add_argument("--target_directory", "-t", required=True, help="Cropped folder directory")
    parser.add_argument("--json_result", "-r", required=True, help="JSON Result name")
    args = parser.parse_args()
    return args

def main():
    # Init Models
    model_fd = FaceDetection(use_cuda=True) # face detection wrapper
    
    args = read_args()
    # Access them
    print("Folder dir:", args.source_directory)
    list_path_images = get_list_images(args.source_directory)
    #print(re.sub(r'\s+', '-', args.name_directory.split('/')[-1]))
    #path_result = f"result_{args.source_directory.split('/')[-2]}.json"
    path_result = f"{args.json_result}.json"
    
    if os.path.exists(path_result):
        with open(path_result, "r") as f:
            data_result = json.load(f)
    else:
        data_result = {}
    
    for id_now,key_now in enumerate(tqdm(list_path_images)):
        if key_now in data_result:
            continue
        else:
            dict_result = {"path":key_now}
            path_img = key_now
            
            # read sample image from file
            try:
                img = cv2.imread(path_img)

                # get face detection result
                dets, angle = model_fd.predict(img)

                # Age
                #out = model_fd.align_single_face(img,dets,angle,outcolor="RGB",crop_size=300)
                img_cropped = model_fd.extract_face(img, dets, angle, task="face-deepfake")
                aligned_img_iso = model_fd.extract_face(img, dets, angle, task="face-iso")
                #img_aligned = out[0]
                # get crop_single_face with loose_factor=1.25
                #img_cropped, _ = model_fd.crop_single_face(img, dets, angle, loose_factor = 1.25)
                # crop the image
                #img_cropped_df, bbox = model_fd.crop_single_face(
                #    img, dets, angle, loose_factor=1.3, crop_size=None,square=True
                #)

                img_name = path_img.split("/")[-1].split(".")[0]+".jpg"

                save_folder = args.target_directory
                save_path_raw = Path(os.path.join(save_folder,"processed",img_name))
                save_path_aligned = Path(os.path.join(save_folder,"iso_aligned",img_name))
                #if save_path_raw.exists():
                #    continue
                
                if img_cropped is not None:
                    save_path_raw.parent.mkdir(parents=True, exist_ok=True)
                    save_path_aligned.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path_raw),img_cropped)
                    cv2.imwrite(str(save_path_aligned),aligned_img_iso)

                #print("HELLO!!")
                #print(f"HERE! {dets[0].shape}")
                dict_result["status"] = "FACE DETECTED"
                dict_result["bbox"] = np.array2string(dets[0][:4], formatter={'float_kind': lambda x: f"{x:.2f}"})
                dict_result["processed"] = str(save_path_raw)
                dict_result["iso_aligned"] = str(save_path_aligned)
                dict_result["label"] = "DEEPFAKE"
                
            
            except KeyboardInterrupt:
                print("Stopped by Ctrl+C")
                raise  # re-raise if you want the program to exit immediately 
                
            except:
                dict_result["status"] = "FACE NOT DETECTED"
                continue
                    
            data_result[key_now] = dict_result

        # Save back
        if id_now%50 == 0:
            with open(path_result, "w") as f:
                print(len(data_result))
                json.dump(data_result, f, indent=4)

if __name__ == "__main__":
    main()