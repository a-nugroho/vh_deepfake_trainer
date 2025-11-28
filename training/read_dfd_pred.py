import pandas as pd
import os
import json
path_source = "/mnt/ssd/workspace/adi/vh_repos_byversion/deepfake-detection/1-1-0/face-production-face-deepfake/evaluation_tools/output_fake-e4s_20251103.txt"
output_json_dir = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/training/collection/attribute"
df_result = pd.read_csv(path_source,sep="|")



folder_source = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/200k_55plus/processed/fake/e4s_20251103_iso0-8"

image_names = df_result["image_name"]
deepfake_scores = df_result["deepfake_score"]
image_paths = [os.path.join(folder_source,i) for i in image_names]

dict_result = {}
for id_img,img_path in enumerate(image_paths):
    dict_result[img_path] = {
            "image_name": image_names[id_img],
            "dfd_score": deepfake_scores[id_img],
            "dfd_version": "1.0.0",
            "label": 1,
            "label_name": "deepfake"
        }

with open(os.path.join(output_json_dir,path_source.split("/")[-1].split(".")[0])+".json", "w") as f:
    json.dump(dict_result, f, indent=4)