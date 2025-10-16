import os
#os.chdir("/mnt/hdd/workspace/adi/verihubs_repos_push/face-production-face-deepfake/")
import sys

sys.path.append(r"/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/utils")  # full path to the folder containing __init__.py
from utils import get_list_images
import matplotlib.pyplot as plt
import argparse
import cv2
from face_detection import FaceDetection
from face_deepfake import DeepfakeDetection

def read_args():
    parser = argparse.ArgumentParser(description="Simple test script")
    # Example arguments
    parser.add_argument("--dir_test", "-d", required=True, help="Directory of images to test")
    args = parser.parse_args()
    return args

fd = FaceDetection()
fdd = DeepfakeDetection()
list_img = 
for img_path in ["evaluation_tools/on-call/2025-09-29/image001.png"]:
    img = cv2.imread(img_path)

    # get face detection result
    dets, ang = fd.predict(img)

    # crop the image
    img_cropped, bbox = fd.crop_single_face(
        img, dets, ang, loose_factor=1.3, crop_size=None, square=True
    )

    #img_cropped, bbox = fd.align_single_face(
    #   img, dets, ang
    #)

    # get deepfake score
    deepfake_score = fdd.predict(img_cropped)
    logits_1 = fdd.engine_1.predict(img_cropped)[0]
    logits_2 = fdd.engine_2.predict(img_cropped)[0]
    logits_3 = fdd.engine_3.predict(img_cropped)[0]
    print(logits_1,logits_2,logits_3)

    # get deepfake decision
    is_deepfake = fdd.classify_predictions(deepfake_score)
    print(f"{img_path.split('/')[-1]}, {deepfake_score}, {is_deepfake}")