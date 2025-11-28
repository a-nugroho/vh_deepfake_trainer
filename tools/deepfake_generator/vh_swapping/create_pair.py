import json
import os
import random
from itertools import chain
from collections import Counter
import copy
import argparse
from tqdm import tqdm
#dataset_name = "taspen_photos"

def read_args():
    parser = argparse.ArgumentParser(description="Template script with named arguments")
    # Example arguments
    parser.add_argument("--dataset_name","-d", required=False, help="Dataset name")
    parser.add_argument("--suffix_output","--s",required=False, default=None)
    args = parser.parse_args()
    return args

def match(v, criteria):
    for key, rule in criteria.items():
        val = v.get(key)
        op = rule["op"]
        target = rule["value"]

        if op == "==":
            if val != target:
                return False
        elif op == "<=":
            if not (isinstance(val, (int, float)) and val <= target):
                return False
        elif op == ">=":
            if not (isinstance(val, (int, float)) and val >= target):
                return False
        else:
            raise ValueError(f"Unsupported operator: {op}")
    return True

def get_pair_key(key_source,dict_collection):

    list_all = [list(d.keys()) for d in dict_collection.values()]
    list_all = list(chain.from_iterable(list_all))
    # Count total frequency
    counter = Counter(list_all)
    max_count = max(counter.values())

    # Get *all* elements with same max frequency
    most_common_elements = [k for k, v in counter.items() if v == max_count]
    if key_source in most_common_elements:
        most_common_elements.remove(key_source)
    return random.choice(most_common_elements) if most_common_elements else None

def get_pair_candidates(list_criteria,data_all):
    dict_collection = {}
    for criteria_now in list_criteria:
        type_criteria = '-'.join(list(criteria_now.keys()))
        dict_collection[type_criteria] = {k: v for k, v in data_all.items() if 
                    match(v,criteria_now)}
    
    return dict_collection

def write_json(file_path,data): 
    directory = os.path.dirname(file_path)  
    
    # Create the directory and any necessary parent directories if they don't exist  
    # exist_ok=True prevents an error if the directory already exists  
    if directory: # Only attempt to create if a directory path is provided  
        os.makedirs(directory, exist_ok=True)  

    with open(file_path, 'w') as f:  
        json.dump(data, f, indent=4) # indent=4 makes the JSON output more readable

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

def merge_nested_existing_only(A, B):
    for k, v in B.items():
        if k in A:
            if isinstance(A[k], dict) and isinstance(v, dict):
                merge_nested_existing_only(A[k], v)  # merge nested dicts
            elif isinstance(A[k], dict):
                # If A[k] is dict but B[k] not dict, skip (prevent overwrite)
                continue
            else:
                # If both are not dicts, skip (don’t overwrite)
                continue
    return A

THRESHOLD_BLUR = 50
THRESHOLD_DARK = 50

def main():

    args = read_args()
    dataset_name = args.dataset_name
    root_dir = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/"
    prefix = "result_"
    path_json = os.path.join(root_dir,f"dataset_assessor/{prefix}{dataset_name}.json")
    dir_att = os.path.join(root_dir,f"dataset_assessor/results_att_fd3-1-0-24Nov/{dataset_name}")
    dir_json =  os.path.join(root_dir,f"deepfake_generator/vh_swapping/collection/")

    if args.suffix_output:
        suffix_output = f"-{args.suffix_output}.json"
    else:
        add_suffix_output = []
        suffix_output = ".json"

    #suffix_output = "-".join(add_suffix_output+[".json"])
    path_output_json = os.path.join(dir_json,f"pair-{dataset_name}{suffix_output}")
    with open(path_json, 'r') as f:
        # Parsing the JSON file into a Python dictionary
        data_source = json.load(f)

    if "200k_live_face_dataset" in path_json:
        data_source_new = {}
        for k, v in data_source.items():
            k_new = k.replace("/processes","")
            data_source_new[k_new] = v
        
        data_source = data_source_new



    dict_att_json = read_att_json(dir_att)
    data_source = merge_nested_dicts(*[data_source,dict_att_json],mode="existing_only")
    data_filtered = {}
    for path_now, d_now in data_source.items():
        # Filter on Blur
        if 'score_blur_face' in d_now and 'score_dark' in d_now and 'head_pose' in d_now:
            if ((d_now['score_blur_face']<THRESHOLD_BLUR) and 
                (d_now['score_dark']<THRESHOLD_DARK) and
                (
                    abs(d_now['head_pose']['yaw'])<10 and
                    abs(d_now['head_pose']['pitch'])<10 and
                    abs(d_now['head_pose']['roll'])<10
                )
            ):
                data_filtered[path_now] = d_now


    data_completed = {}
    swap_pair_list = []

    #print(data_source["../../datasets/deepfake/vh_55plus/raw_images/taspen_photos/13.jpg"])
    #print(dict_att_json["../../datasets/deepfake/vh_55plus/raw_images/taspen_photos/13.jpg"])

    for key_now, value_now in tqdm(data_filtered.items()):
        try:
            pred_age = value_now["pred_age"]
            pred_gender = value_now["pred_gender"]
            list_pair_criteria = [ 
            {"pred_age": {"op": "==", "value": pred_age}},
            {"pred_gender": {"op": "==", "value": pred_gender}}
            ]
            dict_collection = get_pair_candidates(list_pair_criteria,data_filtered)
            swap_pair = get_pair_key(key_now,dict_collection)
            swap_pair_list.append(swap_pair)
            value_now['swap_pair_path'] = swap_pair
            
            for k, v in value_now.items():
                if type(v) == str:
                    value_now[k] = v.replace("../../","/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/")
            key_now = key_now.replace("../../","/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/")
            data_completed[key_now] = {k: value_now[k] for k in ["path","status","swap_pair_path"] if k in value_now} 
            #for subkey_now, subdict_now in data_completed.items():
            #    subkey_now = subkey_now.replace("../../","/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/")
            #    for subsubkey_now, subsubvalue_now in subdict_now.items():
            #        subsubvalue_now = subsubvalue_now.replace("../../","/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/")
            #        subdict_now[subsubkey_now] = subsubvalue_now

            #    data_completed[subkey_now] = subdict_now
        except:
            continue

    write_json(path_output_json,data_completed)

if __name__ == "__main__":
    main()