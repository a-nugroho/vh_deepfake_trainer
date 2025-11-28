import copy
import json
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