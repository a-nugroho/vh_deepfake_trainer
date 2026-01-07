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
import time
# map package name → alternative path
ALT_MODULE_PATHS = {
    "face_deepfake": "/mnt/HDD/workspace/adi/repos/vh_deepfake_trainer/tools/evaluation_tools/face_deepfake",
    "face_detection": "/mnt/HDD/workspace/adi/repos/vh_repos_byversion/face-detection/3-1-0/face-production-face-detection/face_detection"
}

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
            if not init_file.exists():
                return None
            loader = importlib.machinery.SourceFileLoader(fullname, str(init_file))
            return importlib.util.spec_from_loader(fullname, loader, origin=str(init_file))
        return None


def get_all_images_glob(directory_path, extensions=['.png', '.jpg', '.jpeg']):
    """
    Recursively finds all files with specified image extensions using glob.
    """
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory_path, '**', f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=True))
    return image_files

# install our custom finder at the front
finder = AltImportFinder()
if not any(isinstance(f, AltImportFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, finder)

from datetime import datetime

from face_detection import FaceDetection
from face_deepfake.main import softmax
from face_deepfake import DeepfakeDetectionCustom
from face_deepfake.engine_onnx import DeepfakeEngine_Generic

ENGINE_REF = {"convnext":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(384, 384),
        'input_mean':[0.485, 0.456, 0.406],
        'input_std':[0.229, 0.224, 0.225],
        'model_path':'convnext_xlarge_384_in22ft1k_30.pth.onnx'},
    "clip_cip":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(224, 224),
        'input_mean':[0.5, 0.5, 0.5],
        'input_std':[0.5, 0.5, 0.5],
        'model_path':'clip_detector_optimized.onnx'
    },
    "clip_df40pre":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(224, 224),
        'input_mean':[0.5, 0.5, 0.5],
        'input_std':[0.5, 0.5, 0.5],
        'model_path':'clip_large_df40_allff.onnx'
        #'model_path':'clip_df40_large.onnx'
    },
    "clip_eff_vh":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(224, 224),
        'input_mean':[0.5, 0.5, 0.5],
        'input_std':[0.5, 0.5, 0.5],
        'model_path':'effort_vh_2026-01-04_new.onnx'
        #'model_path':'effort_vh_2025-12-24.onnx'
        #'model_path':'effort_vh_1218_2025-12-24.onnx'
        #'model_path':'effort_vh_1218.onnx'

    },
    "clip_eff_vh_2":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(224, 224),
        'input_mean':[0.48145466, 0.4578275, 0.40821073],
        'input_std':[0.26862954, 0.26130258, 0.27577711],
        'model_path':'effort_vh_2025-12-24.onnx'
        #'model_path':'effort_vh_2026-01-01.onnx'
        #'model_path':'effort_vh_1218_2025-12-24.onnx'
        #'model_path':'effort_vh_1218.onnx'
    },
    "clip_eff_vh_3":{'engine_class':DeepfakeEngine_Generic,
        'input_size':(224, 224),
        'input_mean':[0.5, 0.5, 0.5],
        'input_std':[0.5, 0.5, 0.5],
        'model_path':'effort_vh_2026-01-04.onnx'
        #'model_path':'effort_vh_2025-12-24.onnx'
        #'model_path':'effort_vh_1218_2025-12-24.onnx'
        #'model_path':'effort_vh_1218.onnx'

    }
}


timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
# Save all prints to output.txt
sys.stdout = Logger(f"logs/{Path(__file__).stem}o.txt")
fd = FaceDetection(use_cuda=True)
#engine_props = [{'input_size':(224, 224),'input_mean':[0.5, 0.5, 0.5],'input_std':[0.5, 0.5, 0.5],'model_path':'clip_df40_large.onnx'}, # effort
#{'input_size':(384, 384),'input_mean':[0.485, 0.456, 0.406],'input_std':[0.229, 0.224, 0.225],'model_path':'convnext_xlarge_384_in22ft1k_30.pth.onnx'},
##{'input_size':(224, 224),'input_mean':[0.5, 0.5, 0.5],'input_std':[0.5, 0.5, 0.5],'model_path':'clip_detector_optimized.onnx'},
#{'input_size':(224, 224),'input_mean':[0.48145466, 0.4578275, 0.40821073],'input_std':[0.26862954, 0.26130258, 0.27577711],'model_path':'effort_vh_1218.onnx'} # effort-VH
#]

dicts_engine = ENGINE_REF.values()
list_models = list(ENGINE_REF.keys())
fdd = DeepfakeDetectionCustom(device_name="cuda",list_dict_engine=dicts_engine)
parser = argparse.ArgumentParser(description='Process some paths.')
#parser.add_argument('--detector_class', type=str)
#parser.add_argument('--weights_path',nargs="+", type=str)
parser.add_argument("--json_path")
args = parser.parse_args()
num_pred_deepfake = 0
num_pred_real = 0
num_real = 0
num_deepfake = 0

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
#folder_path = args.folder_path
#if args.folder_paths

#for folder_path in folder_paths:
#img_list = get_all_images_glob(folder_path, valid_exts)
json_path = args.json_path
json_file = os.path.join(json_path)
with open(json_file, "r") as f:
    data = json.load(f)
img_list = [k["processed_path"] for k in data.values()]
label_list = [k["label"] for k in data.values()]


with open("_".join([Path(__file__).stem,Path(json_file).stem,"output.txt"]), "w", encoding="utf-8") as f:
    line = '|'.join(["image_name","label","pred_label","deepfake_score","time_process_ms"]
                    +['score_'+i for i in list_models]
                    +['logits_'+i for i in list_models]
                    +['time_ms_'+i for i in list_models])
    f.write(line + "\n")  # write to file

for id_data,img_file in enumerate(img_list):
    start_time =time.time()
    #img_path = os.path.join(os.getcwd(),folder_path,img_file)
    img_path = img_file
    #try:

    # read sample image from file
    img = cv2.imread(img_path)
    if(img is None):
        continue

    # get face detection result
    dets, ang = fd.predict(img)

    if len(dets)==0:
        continue

    img_cropped = fd.extract_face(img, dets, ang, task="face-deepfake")
    #cv2.imwrite(f"pics/{img_path.split('/')[-1]}",img_cropped)
    
    #img_cropped = fd.extract_face(img, dets, ang, task="face-iso")
    #img_cropped = resize_and_center_content_opencv(img_cropped)
    
    # get deepfake score
    deepfake_score, score_all, time_models = fdd.predict(img_cropped, return_all= True, return_time=True)
    #logits_1 = fdd.engine_1.predict(img_cropped)[0]
    #logits_2 = fdd.engine_2.predict(img_cropped)[0]
    #logits_3 = fdd.engine_3.predict(img_cropped)[0]
    
    # get deepfake decision
    is_deepfake = fdd.classify_predictions(deepfake_score)
    end_time = time.time()
    time_ms = (end_time-start_time)*1000
    print(f"Sample {img_file} {'DEEPFAKE' if is_deepfake else 'REAL'}, SCORE: {deepfake_score:.4f}")
    #print(f"Logits 1: {logits_1}, Logits 2: {logits_2}, Logits 3: {logits_3}")
    #except:
    #    print(f"Cant process path {img_path}")
    num_pred_deepfake += int(is_deepfake)
    num_pred_real += int(not is_deepfake)

    #line = f"{img_file}|{'DEEPFAKE' if is_deepfake else 'REAL'}|{deepfake_score}"
    line = '|'.join([img_file,str(label_list[id_data]),'DEEPFAKE' if is_deepfake else 'REAL'
                     ,str(np.round(deepfake_score,5)),str(np.round(time_ms,2))]+
                    [str(np.round(softmax(i)[1].item(),5)) for i in score_all]+
                    [str(np.round(i[0],5))+','+str(np.round(i[1],5)) for i in score_all]+
                    [str(np.round(i,2)) for i in time_models])
    #with open("_".join([Path(__file__).stem,"output.txt"]), "a") as f:
    #    f.write(f"{line}\n")
    with open("_".join([Path(__file__).stem,Path(json_file).stem,"output.txt"]), "a") as f:
        f.write(f"{line}\n")

print(f"Pred Deepfake: {num_pred_deepfake} {100*num_pred_deepfake/(num_pred_deepfake+num_pred_real)}%")
print(f"Pred Real: {num_pred_real} {100*num_pred_real/(num_pred_deepfake+num_pred_real)}%")