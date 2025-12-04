import os
import sys
sys.path.append(r"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/utils")  # full path to the folder containing __init__.py
import json
import re
import cv2
import argparse
from tqdm import tqdm
from age_detection import AgeClassification
from face_detection import FaceDetection
from face_quality_assessment import FaceQualityAssessment 
import csv
import stone
from tools.utils import get_list_images
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


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
            label_name = "MALE"
        elif label_name==1:
            label_name = "FEMALE"
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
    return value_att

def func_tone(data_input):
    result = stone.process(data_input)
    value_att = result["faces"][0]["tone_label"]
    return value_att

def func_gender(data_input):
    faces = app.get(data_input)
    value_att = label_decoding(faces[0]['gender'],"pred_gender")
    return value_att

def result_dumping(root_result, data_result):
    print(len(data_result))
    for att_now, dict_result in data_result.items():
        path_result_att = os.path.join(root_result,att_now+".json")
        os.makedirs(os.path.dirname(path_result_att), exist_ok=True)
        with open(path_result_att, "w") as f:
            json.dump(dict_result, f, indent=4)
        
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
    model_fd = FaceDetection(use_cuda=True) # face detection wrapper
    model_ad = AgeClassification()
    model_qa = FaceQualityAssessment()
    ctx = 0
    global app
    app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
    app.prepare(ctx_id=ctx, det_size=(640,640))

    collection_processor = {"pred_skin_tone":func_tone,
                       "pred_gender":func_gender,
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

    with open(path_json_source, "r") as f:
        data_source = json.load(f)
    

    #if os.path.exists(path_result):
    #    with open(path_result, "r") as f:
    #        data_result = json.load(f)
    #else:
    #    data_result = {}
    list_path_images = list(data_source.keys())
    att_included = ["pred_skin_tone","pred_gender"]
    names_attribute = {j:k for j,k in collection_processor.items() if j in att_included}


    data_result = {}
    for att_now in att_included:
        data_result[att_now] = {}
    root_result = f"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att/{folder_name}"
                
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
            dets, angle = model_fd.predict(img)

            # Age
            #out = model_fd.align_single_face(img,dets,angle,outcolor="RGB",crop_size=300)
            #img_aligned = out[0]
            
            img_aligned = model_fd.extract_face(img,dets,angle, task="face-recognition")
            
            # get crop_single_face with loose_factor=1.25
            img_cropped = model_fd.extract_face(img, dets, angle, task="face-quality")
            

            dict_result_base["status"] = "FACE DETECTED"
        
        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
            raise  # re-raise if you want the program to exit immediately 
        
        except:
            dict_result_base["status"] = "FACE NOT DETECTED"
        
        for name_att, processor_att in names_attribute.items():
            dict_result_att = dict_result_base.copy()
            
            if dict_result_base["status"] == "FACE DETECTED":
                #print(name_att)
                if name_att in ["pred_age","pred_gender"]:
                    value_att = process_attribute(name_att,processor_att,img_aligned)
                elif name_att in ["score_blur_face","score_dark"]:
                    value_att = process_attribute(name_att,processor_att,img_cropped)
                elif name_att in ["pred_skin_tone"]:
                    value_att = process_attribute(name_att,processor_att,path_img)

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

        