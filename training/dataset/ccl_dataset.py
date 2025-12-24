import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
#CyclicalTrain => Main Function
#import tensorflow as tf
import gc
import numpy as np
import os
from collections import defaultdict

def get_difficulty_scores(live_image_paths,deepfake_image_paths,dfd_json_paths):
    result = {}
    L_dict = []
    for path_json in dfd_json_paths:
        with open(path_json,'r') as file:
            data_json = json.load(file)
            L_dict.append(data_json)

    for d in L_dict:
        result.update(d)
    
    data_json = result
    #path_jsons = dfd_json_paths
    #with open(path_json, 'r') as file:
    #    data_json = json.load(file)

    data_json_new = {v["processed_path"]:v for k,v in data_json.items()}
    live_dfty_scores = [data_json_new[i]['dfd_1-0-0'] if 'dfd_1-0-0' in data_json_new[i] 
                            else -100 for i in live_image_paths]
    deepfake_dfty_scores = [1-data_json_new[i]['dfd_1-0-0'] if 'dfd_1-0-0' in data_json_new[i] 
                                else -100 for i in deepfake_image_paths]
    return live_dfty_scores, deepfake_dfty_scores

def build_source_indices(live_sources, deepfake_sources):
    indices_source_live = defaultdict(list)
    for idx, source in enumerate(live_sources):
        indices_source_live[source].append(idx)
    
    indices_source_deepfake = defaultdict(list)
    for idx, source in enumerate(deepfake_sources):
        indices_source_deepfake[source].append(idx)
    return dict(indices_source_live), dict(indices_source_deepfake)

live_image_paths = []
live_labels = []
live_sources = []

deepfake_image_paths = []
deepfake_labels = []
deepfake_sources = []

json_paths = ["test_200k_24nov_merged_train","df40_train_fs_reduced"]
json_folder = '/mnt/ssd/datasets/deepfake/dataset_json/'
for json_path in json_paths:
    if json_folder:
        json_path = os.path.join(json_folder,json_path+'.json')

    with open(json_path, 'r') as f:
        metadata = json.load(f)
    metadata_keys = list(metadata.keys())
    json_name = json_path.split("/")[-1].split(".")[0]
    
    metadata = {d: metadata[d] for d in metadata_keys}
    for img_path, info in metadata.items():
        if info["label"] == 0:
            live_image_paths.append(
                            os.path.join("data", info["processed_path"])
                        )
            live_labels.append(info["label"])
            live_sources.append(json_path)
        else:
            deepfake_image_paths.append(
                os.path.join("data", info["processed_path"])
            )
            
            deepfake_labels.append(info["label"])
            deepfake_sources.append(json_path)

indices_source_live, indices_source_deepfake = build_source_indices(live_sources, deepfake_sources)
dfd_json_paths = [
    '/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/test_200k_24nov_merged_train/dfd_1-0-0.json',
    "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/facebook_dfdc_train_reduced_rev/dfd_1-0-0.json",
    "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/faceforensics++_train/dfd_1-0-0.json",
    "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/df40_train_fs_reduced/dfd_1-0-0.json"
]

live_dfty_scores, deepfake_dfty_scores = get_difficulty_scores(
    live_image_paths, deepfake_image_paths, dfd_json_paths)
    #self.validate_by_difficulty()

def _get_sub_x(x, index):
    Train_x = []
    
    if isinstance(x, list):
        for part_x in x:
            Train_x.append(part_x[index])
    else:
        Train_x = x[index]    
    
    return Train_x

def _get_range_size(x, y):
    if not isinstance(x, list):
        range_size = len(x)
    elif not isinstance(y, list):
        range_size = len(y)
    else:
        range_size = len(x[0])
    return range_size

def get_cycle_sizes(sp, ep, alfa, T):
    S = []
    n = sp
    S.append(n)
    t = 0
    while sum(S) < T:
        if (n == sp) or ((S[t - 1] < S[t]) and (n != ep)):
            n = min((n * (1 / alfa)), ep)
        else:
            n = max((n * (alfa)), sp)
        S.append(n)
        t += 1
    return S

def get_train_data_by_scores(indices_source_live, indices_source_deepfake, scores=None):
    range_size_live = len(indices_source_live)
    pools_source_live = {}
    for src, indices in indices_source_live.items():
        pools_source_live[src] = np.random.choice(indices,size,p=live_dfty_scores,replace=False)
    #pools_source_live = {
    #            src: self.rng.permutation(indices).tolist()
    #            for src, indices in indices_source_live.items()
    #        }
    #index_live = np.random.choice(range(0, range_size), size, p=scores, replace=False)
    #index_fake =
    Train_x = _get_sub_x(x, index)
    Train_y = _get_sub_x(y, index)
    return Train_x, Train_y

def CyclicalTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=1,
    data=None,
):
    total_sample_count = _get_range_size(x, y)
    current_max = 0
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    result_dict = {}
    epochs = len(data_sizes)
    for i in range(epochs):
        sample_count_epoch = int(total_sample_count * data_sizes[i])
        sub_x, sub_y = get_train_data_by_scores(
            x, y, sample_count_epoch, scores=scores / scores.sum()
        )
        
        history = model.fit(
            sub_x,
            sub_y,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=data,
        )
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        
        val_accs.append(current)
        train_accs.append(history.history["accuracy"][0])
        val_losses.append(history.history["val_loss"][0])
        train_losses.append(history.history["loss"][0])        
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    result_dict['accuracy'] = train_accs
    result_dict['loss'] = train_losses
    result_dict['val_accuracy'] = val_accs
    result_dict['val_loss'] = val_losses
    
    return model, current_max, result_dict

def get_categ_ind_loss(model, X, Y, batch_size = 32):

    individual_loss_cal = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.compat.v1.losses.Reduction.NONE)
                
    pred = model.predict(X, batch_size, verbose = 0)
    ind_loss = individual_loss_cal(Y, pred).numpy()
    
    if len(ind_loss.shape) > 1:
        ind_loss = np.sum(ind_loss,axis = 1)
        
    tf.keras.backend.clear_session()
    gc.collect()

    return ind_loss

"""
def get_cycle_sizes(sp, ep, alfa, T):
    S = []
    n = sp
    S.append(n)
    t = 0
    while sum(S) < T:
        if (n == sp) or ((S[t - 1] < S[t]) and (n != ep)):
            n = min((n * (1 / alfa)), ep)
        else:
            n = max((n * (alfa)), sp)
        S.append(n)
        t += 1
    return S

def _get_range_size(x, y):
    if not isinstance(x, list):
        range_size = len(x)
    elif not isinstance(y, list):
        range_size = len(y)
    else:
        range_size = len(x[0])
    return range_size

def _get_sub_x(x, index):
    Train_x = []
    
    if isinstance(x, list):
        for part_x in x:
            Train_x.append(part_x[index])
    else:
        Train_x = x[index]    
    
    return Train_x

# Called when initiating training object
start_percent = 0.25
end_percent = 1
multiplier = 0.50
#data_sizes = get_cycle_sizes(start_percent, end_percent, multiplier, EPOCHS),

def get_train_data_by_scores(x, y, size, scores=None):
    range_size = _get_range_size(x, y)
    index = np.random.choice(range(0, range_size), size, p=scores, replace=False)
    Train_x = _get_sub_x(x, index)
    Train_y = _get_sub_x(y, index)
    return Train_x, Train_y

"""