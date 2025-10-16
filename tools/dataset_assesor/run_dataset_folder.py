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
from face_deepfake import DeepfakeDetection
import csv
from utils import get_list_images
root_json = "/mnt/ssd/datasets/deepfake/dataset_json"

def read_data(root_json, name_directory):
    path_json = os.path.join(root_json,name_directory+".json")
    with open(path_json, 'r') as file:
        data_json = json.load(file)
    return data_json

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--name_directory", "-d", required=True, help="Dataset folder")
    args = parser.parse_args()
    return args

def main():
    # Init Models
    model_fd = FaceDetection(use_cuda=True) # face detection wrapper
    model_ad = AgeClassification()
    model_qa = FaceQualityAssessment()
    model_fdd = DeepfakeDetection(device_name="cuda")

    
    args = read_args()
    # Access them
    print("Folder dir:", args.name_directory)
    list_path_images = get_list_images(args.name_directory)
    #print(re.sub(r'\s+', '-', args.name_directory.split('/')[-1]))
    path_result = f"result_{args.name_directory.split('/')[-2]}.json"

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
                out = model_fd.align_single_face(img,dets,angle,outcolor="RGB",crop_size=300)
                img_aligned = out[0]
                # get crop_single_face with loose_factor=1.25
                img_cropped, _ = model_fd.crop_single_face(img, dets, angle, loose_factor = 1.25)
                # crop the image
                img_cropped_df, bbox = model_fd.crop_single_face(
                    img, dets, angle, loose_factor=1.3, crop_size=None,square=True
                )

                dict_result["status"] = "FACE DETECTED"
            except KeyboardInterrupt:
                print("Stopped by Ctrl+C")
                raise  # re-raise if you want the program to exit immediately 
            
            except:
                dict_result["status"] = "FACE NOT DETECTED"
                #continue
                
            try:
                # Age classification
                age_class = model_ad.predict(img_aligned)
                dict_result["pred_age"] = age_class
            except:
                dict_result["pred_age"] = "UNKNOWN"
            
            
            try:
            # get blur score
                blur_score_face = model_qa.blur.blur_score_selfie(img_cropped)
                dict_result["score_blur_face"] = blur_score_face
            except:
                dict_result["score_blur_face"] = -100
            
            # get blur detection
            #is_blur_face = model_qa.blur.blur_detection(blur_score_face)

            # get dark score
            try:
                dark_score = model_qa.dark.contrast_score_selfie(img_cropped)
                dict_result["score_dark"] = dark_score
            except:
                dict_result["score_dark"] = -100
            
            # get dark detection
            #is_dark = model_qa.dark.contrast_detection(dark_score)

            # get grayscale score
            #raw_score = model_qa.grayscale.grayscale_score_selfie_raw(img_cropped)

            # get the grayscale detection
            #is_gray = model_qa.grayscale.grayscale_detection(raw_score)

            # get the blur score
            try:
                blur_score_img = model_qa.image_quality.quality_score_selfie(img)
                dict_result["score_blur_img"] = blur_score_img
            except:
                dict_result["score_blur_img"] = -100
                
            # get the blur detection
            #is_blur_img = model_qa.image_quality.blur_detection(blur_score_img)

            # get deepfake score
            try:
                deepfake_score = model_fdd.predict(img_cropped_df)
                dict_result["score_dfd1-0-0"] = deepfake_score
            except:
                dict_result["score_dfd1-0-0"] = -1

            # get deepfake decision
            #is_deepfake = model_fdd.classify_predictions(deepfake_score)

            #print(f"{is_deepfake}")
            # Load existing data if file exists
            data_result[key_now] = dict_result

        # Save back
        if id_now%50 == 0:
            with open(path_result, "w") as f:
                print(len(data_result))
                json.dump(data_result, f, indent=4)

if __name__ == "__main__":
    main()

        