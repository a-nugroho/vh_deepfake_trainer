import json, os

"""
json_face_swapped = ["result_e4s_20251103-strictpose.json",
"result_blendface-strictpose.json",
"result_inswapper-strictpose.json",
"result_mobilefaceswap-strictpose.json",
"result_reswapper-strictpose.json"]
"""
json_face_swapped = [
"200k_24nov_e4s.json",
"200k_24nov_blendface.json",
"200k_24nov_inswapper.json",
"200k_24nov_mobilefaceswap.json",
"200k_24nov_reswapper.json"
]
json_face_real = "200k_live_face_dataset.json"
dir_json = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/datasets/deepfake/dataset_json"
list_all = []
for json_now in json_face_swapped:
    json_path = os.path.join(dir_json,json_now)
    with open(json_path, "r") as f:
        dict_att = json.load(f)

    #38368_verify_67dfb808-1e6c-47d6-b5ca-d23ae43dea20_11e03acf-3c0b-4e3d-96b1-6333e424a80f_original_to_169604_verify_90919c30-5031-4b92-a0cd-bb8e565bd660_4db71cc5-dbf6-4b36-9391-c00364c84101_original.jpg
    list_keys_1 = [i["image_path"].split("/")[-1].split("_to_")[0] for i in dict_att.values()]
    list_keys_2 = [i["image_path"].split("/")[-1].split("_to_")[1] for i in dict_att.values()]
    list_all.extend(list_keys_1+list_keys_2)
    list_all = list(set(list_all))


json_facereal_path = os.path.join(dir_json, json_face_real)
with open(json_facereal_path, "r") as f:
    dict_att = json.load(f)

print(list_all[0])
dict_selected = {k:v for k,v in dict_att.items() if k.split("/")[-1].split(".")[0] in list_all}

output_json = "200k_live_face_24nov.json"
with open(os.path.join(dir_json,output_json), "w") as f:
    json.dump(dict_selected, f, indent=4)