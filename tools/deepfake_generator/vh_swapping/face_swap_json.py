import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from tqdm import tqdm
import argparse
import json
assert insightface.__version__>='0.7'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--output_dir", type=str, default="examples/output.png")
    parser.add_argument("-j","--source_json", type=str)
    args = parser.parse_args()
    return args

def get_source_target_list(path_json):
    list_pair_path = []
    with open(path_json, 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data = json.load(f)

    for key, value in data.items():
        list_pair_path.append([key, value["swap_pair_path"]])
    
    return list_pair_path

def swap_pair(swapper, source_path, target_path):
    img = cv2.imread(source_path)
    if img is None:
        return None

    source_faces = app.get(img)
    
    img = cv2.imread(target_path)
    if img is None:
        return None
    
    target_faces = app.get(img)
    
    if (len(source_faces)>0 and len(target_faces)>0):
        _img = swapper.get(img, target_faces[0], source_faces[0], paste_back=True)
    else:
        _img = None
    return _img
    
if __name__ == '__main__':
    args = get_args()
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    model_name = "inswapper_128"
    swapper = insightface.model_zoo.get_model(f'{model_name}.onnx', download=True, download_zip=True)
    
    path_json = args.source_json
    list_pair_path = get_source_target_list(path_json)
    for i, ([source_path, target_path]) in enumerate(tqdm(list_pair_path)):
        
        res = swap_pair(swapper, source_path, target_path)
        if res is not None:
            name_output = '_to_'.join([os.path.basename(source_path).split(".")[0],  os.path.basename(target_path).split(".")[0]+'.jpg'])
            os.makedirs(args.output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, name_output), res)
            #res.save(os.path.join(args.output_dir, name_output))
    """
    img = ins_get_image('t1')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    assert len(faces)==6
    source_face = faces[2]
    res = img.copy()
    
    # debuging
    #for face in faces:
    #    res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("./t1_swapped2.jpg", res)
    """
