import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm
import csv
import datetime
import random
import cv2
from PIL import Image
import yaml
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from detectors import DETECTOR
from dataset import DeepFakeDataset,ProportionalStratifiedBatchSampler
# Calculate ROC AUC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# Dataset Definition
import dataset.utils.aug.low_quality as low_quality
import dataset.utils.aug.crop as crop
import dataset.utils.aug.border as border

from logger import create_logger, RankFilter


# -------------------------------
# Auxilliary Functions
# -------------------------------

def preprocess_batch(batch, size=(224, 224),
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """
    Resize and normalize a batch of images.

    Args:
        batch (Tensor): Input tensor of shape [B, C, H, W].
        size (tuple): Target (H, W).
        mean (tuple): Per-channel mean for normalization.
        std (tuple): Per-channel std for normalization.

    Returns:
        Tensor: Preprocessed batch [B, C, H_new, W_new].
    """
    # Resize
    batch = F.interpolate(batch, size=size, mode="bilinear", align_corners=False)

    # Normalize
    mean = torch.tensor(mean, device=batch.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=batch.device).view(1, -1, 1, 1)
    batch = (batch - mean) / std

    return batch

def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer

def generate_feat(model, data_dict):
    with torch.no_grad():
        feat = model.features(data_dict)
        if len(feat.shape)==4:
            feat = feat.mean(dim=[2,3])
        return feat

# -------------------------------
# 1. Load argument and parser
# -------------------------------
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='config/detector/cnn_ens.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.detector_path, 'r') as f:
    config = yaml.safe_load(f)

with open('config/vhubs_train_config.yaml', 'r') as f:
    config2 = yaml.safe_load(f)
if 'label_dict' in config:
    config2['label_dict']=config['label_dict']
config.update(config2)
config['local_rank']=args.local_rank
if config['dry_run']:
    config['nEpochs'] = 0
    config['save_feat']=False
# If arguments are provided, they will overwrite the yaml settings
if args.train_dataset:
    config['train_dataset'] = args.train_dataset
if args.test_dataset:
    config['test_dataset'] = args.test_dataset
config['save_ckpt'] = args.save_ckpt
config['save_feat'] = args.save_feat

# -------------------------------
# 2. Set Config and Hyperparam
# -------------------------------

prep_size = [(224,224),(384,384),(224,224)]
prep_mean = [[0.48145466, 0.4578275, 0.40821073], [0.485, 0.456, 0.406], [0.5, 0.5, 0.5]]
prep_std =  [[0.26862954,0.26130258,0.27577711], [0.229, 0.224, 0.225], [0.5, 0.5, 0.5]]

model_list = ["effort_2025-07-08","dfgc_convnext","clip_ciplab"]

config_base_models =[
        "config/detector/effort_vh.yaml",
        "config/detector/convnext_xl.yaml",
        "config/detector/clip_ciplab.yaml"
    ]

weight_path_list =[
        "../logs/training/effort_2025-07-08-12-51-27/train/200k_live_face_dataset,facebook_dfdc_train_reduced,df40_train_reduced,reswapper_v2_train,reswapper_v2_val,inswapper,ffhq/ckpt_latest.pth",
        "../pretrained/convnext_xlarge_384_in22ft1k_30.pth",
        "../pretrained/clip_ciplab_best.pth"
    ]
    
#transform_effort = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                std=[0.26862954,0.26130258,0.27577711])
#normalize_convnext = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#normalize_clip = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#transform_totensor = transforms.Compose(
#    [
#        transforms.Resize((224, 224)),
#        transforms.ToTensor(),
#    ]
#)


# -------------------------------
# 3. Dataset & Dataloader
# -------------------------------
class StratifiedSourceDataset(Dataset):
    def __init__(self, json_paths, json_folder=None, train=True, ssl=False, transform=None):
        
        # Dummy data
        self.data = []
        self.live_image_paths = []
        self.live_labels = []
        self.live_sources = []
        self.deepfake_image_paths = []
        self.deepfake_labels = []
        self.deepfake_sources = []

        for json_path in json_paths:
            if json_folder:
                json_path = os.path.join(json_folder,json_path+'.json')

            with open(json_path, 'r') as f:
                metadata = json.load(f)


            # HERE, need to include method to subsample
            for img_path, info in metadata.items():
                self.process_data(info, img_path, json_path)
                
           
        self.live_source_indices, self.deepfake_source_indices = self._build_source_indices()

        self.live_len = len(self.live_image_paths)
        self.deepfake_len = len(self.deepfake_image_paths)

        self.train = train
        self.ssl = ssl

        if transform is None:
            if not train:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop((224,224)),
                ])

                to_tensor = [transforms.ToTensor()]
                
                
            else:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    #low_quality.RandomDownUpSampler(p=0.25,downsampling_range=(0.33,0.8)),
                    #low_quality.SimulateLowQuality(p=0.5),
                    transforms.RandomResizedCrop(size=(224, 224), scale=(0.66, 1.0)),
                    #border.RandomBlackBorderFixedSizeSquare(max_border_ratio=0.3, p=0.5),
                ])

                to_tensor = [transforms.ToTensor()]

                
            self.transform = transforms.Compose(self.transform.transforms + to_tensor)
        else:
            self.transform = transform

        self.data_dict = {
            'image': self.live_image_paths+self.deepfake_image_paths, 
            'label': self.live_labels+self.deepfake_labels, 
        }
        #print(self.transform)
        
    def _build_source_indices(self):
        live_source_indices = defaultdict(list)
        for idx, source in enumerate(self.live_sources):
            live_source_indices[source].append(idx)
        
        deepfake_source_indices = defaultdict(list)
        for idx, source in enumerate(self.deepfake_sources):
            deepfake_source_indices[source].append(idx)
        return dict(live_source_indices), dict(deepfake_source_indices)

    def __len__(self):
        return max(self.live_len, self.deepfake_len)
    
    def __min_len__(self):
        return min(self.live_len, self.deepfake_len)
    
    def load_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image at path {image_path}: {e}")
            return None
        
    def __getitem__(self, tuple_idxs):
        #print(tuple_idxs)
        live_images = []
        deepfake_images = []
        live_labels = []
        deepfake_labels = []
        #for idx_live, idx_deepfake in tuple_idxs:
        idx_live, idx_deepfake = tuple_idxs
        while True:
            try:
                # Get live image and its label
                live_idx = idx_live % self.live_len
                live_image = self.load_image(
                    self.live_image_paths[live_idx]
                )
                live_label = self.live_labels[live_idx]

                # Get deepfake image and its label
                deepfake_idx = idx_deepfake % self.deepfake_len
                deepfake_image = self.load_image(
                    self.deepfake_image_paths[deepfake_idx]
                )
                deepfake_label = self.deepfake_labels[deepfake_idx]
                return self.transform(live_image), self.transform(deepfake_image), live_label, deepfake_label
                
                
            except Exception as e:
                print(f"Error loading image at index {idx_live}: {e}")
                idx_live = (idx_live + 1) % self.live_len
        
        #return torch.stack(live_images), torch.stack(deepfake_images), torch.stack(live_labels), torch.stack(deepfake_labels)
    
    def process_data(self, info, img_path, json_path):
        """
        Processes training data and appends paths and labels for both live and deepfake images.

        Args:
            info (dict): Metadata of the image.
            img_path (str): Path to the image in the JSON metadata.
        """
        if info["label"] == 0:
            self.live_image_paths.append(
                os.path.join("data", info["processed_path"])
            )
            self.live_labels.append(info["label"])
            self.live_sources.append(json_path)

        else:
            self.deepfake_image_paths.append(
                os.path.join("data", info["processed_path"])
            )
            
            self.deepfake_labels.append(info["label"])
            self.deepfake_sources.append(json_path)

def get_train_loader(config,):
    train_set = StratifiedSourceDataset(
            config['train_dataset'],json_folder=config['dataset_json_folder'], train=True
    )
    live_source_indices_dict = train_set.live_source_indices
    str_live = []
    for k,v in live_source_indices_dict.items():
        str_live.append(f"{k} {len(v)}")

    str_live = '|'.join(str_live)
    logger.info(str_live)

    deepfake_source_indices_dict = train_set.deepfake_source_indices
    str_deepfake = []
    for k,v in deepfake_source_indices_dict.items():
        str_deepfake.append(f"{k} {len(v)}")
    
    str_deepfake = '|'.join(str_deepfake)
    logger.info(str_deepfake)
    train_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                num_workers=int(config['workers']),
                #batch_sampler= MultiEpochStratifiedSampler(train_set, batch_size=config['train_batchSize'],epochs_per_cycle=200)
                batch_sampler= ProportionalStratifiedBatchSampler(train_set, batch_size=config['train_batchSize'])
                )
    
    return train_loader

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        if 'dataset_type' in config and config['dataset_type'] == 'verihubs':
            test_set = DeepFakeDataset(
                test_name,json_folder=config['dataset_json_folder'], train=False
            )
        
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders

def add_prefix_to_state_dict(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"{prefix}.{key}"
        new_state_dict[new_key] = value
    return new_state_dict

def get_method(row):
    dataset = row['dataset']
    img_path = str(row['image_path'])
    method = None
    if 'reswapper' in dataset.lower():
        method = 'reswapperv2'
        return method
    elif 'ffhq' in dataset.lower():
        method = 'ffhq'
        return method
    else:
        folder_lvone = img_path.split('/')[7]
        if 'indonesian_deepfake' in dataset.lower():
            if folder_lvone == 'live':
                #method = ';'.join([folder_lvone,"verification"])
                method = "verification"
            elif folder_lvone == 'deepfake':
                #method = ';'.join([folder_lvone,img_path.split('/')[8]])
                method = img_path.split('/')[8]
            return method
        elif 'df40' in dataset.lower():
            #if folder_lvone == 'live':
            #    #method = ';'.join([folder_lvone,"verification"])
            #    method = "verification"
            #elif folder_lvone == 'deepfake':
            #    #method = ';'.join([folder_lvone,img_path.split('/')[8]])
            method = img_path.split('/')[7]
            return method
        elif 'facebook_dfdc' in dataset.lower():
            #if folder_lvone == 'live':
            #    #method = ';'.join([folder_lvone,"verification"])
            #    method = "verification"
            #elif folder_lvone == 'deepfake':
            #    #method = ';'.join([folder_lvone,img_path.split('/')[8]])
            method = 'facebook_dfdc'
            return method
        elif 'inswapper' in dataset.lower():
            method = img_path.split('/')[7]  
            return method
        elif 'vh_production_face_deepfake_v1_eval' in dataset.lower():
            method = img_path.split('/')[7]
            return method
        elif 'bing_crawl' in dataset.lower():
            method = 'bing_crawl'
            return method
        elif 'client' in dataset.lower():
            method = 'client'
            return method
        elif 'faceforensics++' in dataset.lower():
            method = img_path.split('/')[7]
            return method

# -------------------------------
# 4. Model
# -------------------------------
class HybridAttentionEnsemble(nn.Module):
    def __init__(self, dim1, dim2, dim3, hidden_dim=256, num_heads=4, num_layers=2, 
                 dropout=0.1, noise_std=0.05, expert_dropout_p=0.2, feature_dropout_p=0.1, init_dropout_p=0, use_mixup=False):
        super().__init__()

        self.noise_std = noise_std

        # --- Stage 1: project features to common hidden dim
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.proj3 = nn.Linear(dim3, hidden_dim)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Unique learnable queries for each expert
        self.query_class = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Final classifier
        self.fc = nn.Linear(hidden_dim, 2)

        # Augmentation params
        self.expert_dropout_p = expert_dropout_p
        self.feature_dropout_p = feature_dropout_p
        self.init_dropout = nn.Dropout(init_dropout_p)
        self.feature_dropout = nn.Dropout(feature_dropout_p)
        self.use_mixup = use_mixup

        # Store input dims for splitting
        self.dims = [dim1, dim2, dim3]

    """
    def add_noise(self, x):
        #Apply Gaussian noise + feature dropout + optional shuffling
        if self.training:
            # Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

            # Feature dropout (randomly zero out some feature dims)
            if self.feature_dropout_p > 0:
                mask = (torch.rand_like(x) > self.feature_dropout).float()
                x = x * mask

            # Random shuffle of feature dimensions (stochastic regularization)
            if random.random() < 0.1:  # 10% chance
                idx = torch.randperm(x.size(-1), device=x.device)
                x = x[:, idx]

        return x
    """
    
    def forward(self, feats, target=None):
        feat1, feat2, feat3 = torch.split(feats, self.dims, dim=1)
        B = feat1.size(0)

        feat1 = self.init_dropout(feat1)
        feat2 = self.init_dropout(feat2)
        feat3 = self.init_dropout(feat3)
        # --- Stage 1: projection
        f1 = F.relu(self.proj1(feat1)).unsqueeze(1)  # [B, 1, H]
        f2 = F.relu(self.proj2(feat2)).unsqueeze(1)
        f3 = F.relu(self.proj3(feat3)).unsqueeze(1)

        # --- Expert dropout (drop entire expert stream)
        #if self.training and self.expert_dropout_p > 0:
        #    if random.random() < self.expert_dropout_p: f1 = torch.zeros_like(f1)
        #    if random.random() < self.expert_dropout_p: f2 = torch.zeros_like(f2)
        #    if random.random() < self.expert_dropout_p: f3 = torch.zeros_like(f3)

        if self.training and self.expert_dropout_p > 0:
            if random.random() < self.expert_dropout_p:
                # Drop exactly one expert (uniform over 3 choices)
                drop_idx = torch.randint(0, 3, (1,)).item()
                if drop_idx == 0:
                    f1 = torch.zeros_like(f1)
                elif drop_idx == 1:
                    f2 = torch.zeros_like(f2)
                else:
                    f3 = torch.zeros_like(f3)
            # else: keep all experts

        # --- Mixup (optional)
        if self.training and self.use_mixup and target is not None:
            lam = np.random.beta(0.4, 0.4)
            perm = torch.randperm(B)
            f1 = lam * f1 + (1 - lam) * f1[perm]
            f2 = lam * f2 + (1 - lam) * f2[perm]
            f3 = lam * f3 + (1 - lam) * f3[perm]
            target = lam * target + (1 - lam) * target[perm]  # soft labels

        
        # Concatenate refined features
        refined = torch.cat([f1, f2, f3], dim=1)  # [B, 3, H]
        refined = self.feature_dropout(refined)         # feature dropout

        # --- Cross-feature fusion
        fusion_q = self.query_class.expand(B, -1, -1)   # [B, 1, H]
        out1 = self.shared_encoder(torch.cat([fusion_q, refined], dim=1))[:, 0:1, :]
        out1 = out1 + f1 + f2 + f3 # Residual direct line
        # --- Classifier
        logits = self.fc(out1.squeeze(1))  # [B, 2]

        if self.training and self.use_mixup and target is not None:
            return logits, target  # return mixed targets for loss
        return logits
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_base_model(config, weights_path):
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config).to(device)

    # Load standard model weights
    checkpoint = torch.load(weights_path, map_location=device)
        
    if model_class == "effort":

        if checkpoint.get('state_dict'):
            checkpoint['state_dict'] = add_prefix_to_state_dict(checkpoint['state_dict'], "backbone.net")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
        
    if model_class == "convnext":
        if checkpoint.get('state_dict'):
            print("HERE1!")
            checkpoint['state_dict'] = add_prefix_to_state_dict(checkpoint['state_dict'], "backbone.net")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            checkpoint = add_prefix_to_state_dict(checkpoint, "backbone.net")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
    
    if model_class == "clip":
        if config["direct_load"]:
            # remove module.
            new_state_dict = {}
            for key, value in checkpoint.items():
                new_key = key
                new_state_dict[new_key] = value
            
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(missing_keys)
    print(f"Loaded {config['model_name']} model weights successfully.")

    return model

# --------------------------
# 5. Training & Validation Loops
# --------------------------

def train_one_epoch(epoch, model_ens, base_models, loader, criterion, optimizer, device, save_iter=200):
    model_ens.train()
    for model_now in base_models:
        model_now.eval()
    
    acc_val = 0.0
    f1_val = 0.0
    avg_val_loss = 9999
    running_loss, correct, total = 0.0, 0, 0
    last_idx = len(loader) - 1
        
    for batch_idx, (input_live, input_fake, target_live, target_fake, *additional_input) in tqdm(enumerate(loader), total=len(loader)):
        iteration = batch_idx
        last_batch = batch_idx == last_idx
        input_live, target_live = input_live.cuda(), target_live.cuda()
        input_fake, target_fake = input_fake.cuda(), target_fake.cuda()
        
        input = torch.cat((input_live,input_fake))
        target = torch.cat((target_live,target_fake))
        additional_input = [i.cuda() for i in additional_input]

        # 2. Shuffle using a random permutation
        perm = torch.randperm(input.size(0))     # shuffled indices
        input = input[perm]
        target = target[perm]

        data_dict = {}
        data_dict['image'] = input
        data_dict['label'] = target
            
        optimizer.zero_grad()
            
        list_feat =  []
        for k,model_now in enumerate(base_models):
            data_dict_now = data_dict
            data_dict_now["image"] = preprocess_batch(data_dict_now["image"],size=prep_size[k],mean=prep_mean[k],std=prep_std[k])
            list_feat.append(generate_feat(model_now, data_dict_now))
        
        feats = torch.cat(list_feat,-1)

        labels = data_dict["label"]
        logits = model_ens(feats, target=F.one_hot(labels, num_classes=2).float() if model_ens.use_mixup else None)
        targets = labels
        loss = criterion(logits, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()


        #running_loss += loss.item() * feats.size(0)
        running_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        if iteration%save_iter==0:
            logger.info(f"===> Epoch[{epoch}] running avg loss: {running_loss/(iteration+1)} running train acc: {correct/total}")
            save_ckpt(model_ens,'train', ','.join(config['train_dataset']), f"{epoch}-{iteration}",save_latest=True)
                

    avg_loss = running_loss / (batch_idx+1)
    acc = correct / total
    return avg_loss, acc


# --------------------------
# 6. Save & Load Checkpoints
# --------------------------
def save_checkpoint(model, optimizer, epoch, config, val_acc):
    os.makedirs(config.save_dir, exist_ok=True)
    filename = os.path.join(config.save_dir, f"epoch{epoch}_acc{val_acc:.4f}.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc
    }, filename)
    print(f"✅ Saved checkpoint: {filename}")

def save_ckpt(model, phase, dataset_key,ckpt_info=None,save_latest=False):
    save_dir = os.path.join(logger_path, phase, dataset_key)
    os.makedirs(save_dir, exist_ok=True)
    if save_latest:
        ckpt_name = f"ckpt_latest.pth"
    else:
        ckpt_name = f"ckpt_best.pth"
    save_path = os.path.join(save_dir, ckpt_name)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")


# --------------------------
# 7. Main Training Loop
# --------------------------

def main():
    global logger
    global logger_path
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    # prepare the training data loader
    train_loader = get_train_loader(config)
    
    # prepare the testing data loader
    test_loaders = prepare_testing_data(config)

    model_ens = HybridAttentionEnsemble(1024, 2048, 768,hidden_dim=128,expert_dropout_p=0.33,
                            num_layers=4,dropout=0.1,feature_dropout_p=0.1, init_dropout_p=0,use_mixup=False).cuda()
    model_ens.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = choose_optimizer(model_ens,config)
    
    base_models = []
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for id_c,config_base in enumerate(config_base_models):
        with open(config_base, 'r') as f:
            config_base_models[id_c] = yaml.safe_load(f)
        
        model_now = load_base_model(config_base_models[id_c],weight_path_list[id_c])
        for param in model_now.parameters():
            param.requires_grad = False

        base_models.append(model_now)
    
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        train_loss, train_acc = train_one_epoch(epoch, model_ens, base_models, train_loader, criterion, optimizer, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        #history["val_loss"].append(val_loss)
        #history["val_acc"].append(val_acc)

        # Periodic checkpoint saving
        #if epoch % config.save_every == 0:
        #    save_checkpoint(model, optimizer, epoch, config, val_acc)
        #logger.info(f"===> Epoch[{epoch}] Train loss: {train_loss} Train acc: {train_acc}")
        #save_ckpt(model_ens,'train', ','.join(config['train_dataset']), f"{epoch}",save_latest=True)
    # Final test evaluation
    #test_loss, test_acc = validate(model, test_loader, criterion, config.device)
    #print(f"🎯 Test Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    


if __name__ == '__main__':
    main()
    
"""
    for detector_now in detector_path_list:
        with open(detector_now, "r") as f:
            config = yaml.safe_load(f)
        config_list.append(config)
    
    for k in range(len(config_list)):
        model_collection.append(load_base_model(
            config_list[k],
            weights_path=weight_path_list[k])
            )
        
    
    model(torch.randn(1,1024+2048+768).cuda())
    df_master_train = create_dataset(train_dataset_list)
    df_master_test = create_dataset(test_dataset_list)

    print(model_list)
    train_dataset_all = FeatureDataset(dataset_list=train_dataset_list,df_master=df_master_train,model_list = model_list)
    test_dataset_all = FeatureDataset(dataset_list=test_dataset_list,df_master=df_master_test,model_list = model_list)
    train_loader = DataLoader(train_dataset_all, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(test_dataset_all, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)
    #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    #model = EnsembleNet()
    #model = TransformerEnsemble(1024, 2048, 768,hidden_dim=16)
     # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    print(f"Total learnable parameters: {count_parameters(model):,}")
    #model = ens_net.cuda()
    
    model_train = train(model_collection,model,train_loader, val_loader, num_epochs=1000, lr=2e-3, patience=100, device="cuda")
    

def train(model_collection,model,train_loader, val_loader, num_epochs=5, lr=1e-4, patience=10, device="cuda"):
    # dataset + dataloader
    #dataset = FeatureDataset(root_dir)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # model
    #model = EnsembleNet(input_dims).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
    weight_decay=1e-2)
    #criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)  # we'll apply weights manually
    criterion = nn.CrossEntropyLoss(reduction="none")
    acc_val = 0.0
    f1_val = 0.0
    avg_val_loss = 9999
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = -1
    best_val_loss = float("inf")
    patience_counter = 0

    # training
    csv_file ="training_log.csv" 
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(["Epoch", "Train Loss", "Train F1", "Last Val Loss", "Last Val Acc", "Last Val F1"])
    
    for epoch in range(num_epochs):
        all_train_preds, all_train_labels, all_train_datasets = [], [], []
            
        model.train()
        for model_now in model_collection:
            model_now.eval()
        
        epoch_loss = 0.0
        #for feats, labels, weights, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        for images, labels, weights, dataset_names in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data_dict = {"image": images.to(device), "label": labels.to(device)}
        
            optimizer.zero_grad()
            feat_collection = []
        
            for model_now in model_collection:
                feat_now = generate_feat(model_now, data_dict).cpu()
                feat_collection.append(feat_now)
            
            feats = torch.cat(feat_collection,dim=-1)
            #logits, targets = model(feats, target=F.one_hot(labels, num_classes=2).float() if model.use_mixup else None)
            logits = model(feats, target=F.one_hot(labels, num_classes=2).float() if model.use_mixup else None)
            targets = labels
            # for cross-entropy
            loss = criterion(logits, targets) * weights
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            #true = labels.argmax(dim=1).cpu().numpy()
            true = labels.cpu().numpy()
                    
            all_train_preds.extend(preds)
            all_train_labels.extend(true)
            all_train_datasets.extend(dataset_names)

        
        avg_train_loss = epoch_loss / len(train_loader)
        acc_train = accuracy_score(all_train_labels, all_train_preds)
        f1_train  = f1_score(all_train_labels, all_train_preds, average="weighted")
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={acc_train:.4f}, Train F1={f1_train:.4f}")
        # write a new row each epoch
        
        # Build a DataFrame for convenience
        df_result = pd.DataFrame({
            "labels": all_train_labels,
            "preds": all_train_preds,
            "dataset": all_train_datasets
        })
        # Compute accuracy per subgroup
        #acc_per_group = df_result.groupby("dataset").apply(lambda g: accuracy_score(g["labels"], g["preds"]))

        #print(acc_per_group.to_string())

        # ----- Validation every 5 epochs -----
        if epoch % 5 == 0:
            model.eval()
            all_preds, all_labels, all_datasets = [], [], []
            group_preds, group_labels = {},{}
            val_loss = 0.0
            with torch.no_grad():
                for feats, labels, weights, dataset_names in val_loader:
                    feats, labels, weights = feats.to(device), labels.to(device), weights.to(device)
                    logits = model(feats)
                    
                    # for cross-entropy
                    loss = criterion(logits, labels)
                    loss = loss.mean()
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    true = labels.cpu().numpy()
                    
                    val_loss += loss.item()
                    all_preds.extend(preds)
                    all_labels.extend(true)
                    all_datasets.extend(dataset_names)

            avg_val_loss = val_loss / len(val_loader)
            acc_val = accuracy_score(all_labels, all_preds)
            f1_val  = f1_score(all_labels, all_preds, average="weighted")

            # Build a DataFrame for convenience
            df_result = pd.DataFrame({
                "labels": all_labels,
                "preds": all_preds,
                "dataset": all_datasets
            })
            
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={acc_val:.4f}, Val F1={f1_val:.4f}")

            # Compute accuracy per subgroup
            acc_per_group = df_result.groupby("dataset").apply(lambda g: accuracy_score(g["labels"], g["preds"]))

            print(acc_per_group)
            # ----- Early stopping check -----
            stop_flag = False
            if f1_val == 1.0 and acc_val == 1.0:  
                # if metrics already perfect → monitor validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_ensemble_m05.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        stop_flag = True
            else:
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_acc = acc_val
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), "best_ensemble_m05.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch} (Best F1={best_f1:.4f} at epoch {best_epoch})")
                        break

            if stop_flag:
                print(f"Early stopping at epoch {epoch}. Best Val F1={best_f1:.4f}, Val Acc={best_acc:.4f}, Best Val Loss={best_val_loss:.4f}")
                break
        #print(f"Epoch {epoch+1}: loss={epoch_loss/len(loader):.4f}")
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{f1_train:.4f}", f"{avg_val_loss:.4f}", f"{acc_val:.4f}", f"{f1_val:.4f}"])
        
        #break
    print(f"Training complete. Best Val Acc={best_acc:.4f}, Best Val F1={best_f1:.4f} (epoch {best_epoch})")
    return model
"""

"""
transform_clip = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

transform_convnext = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
"""
#model1_result_folder = "training/results_evaluation/effort_2025-07-08"
#model2_result_folder = "training/results_evaluation/dfgc_convnext"
#model3_result_folder = "training/results_evaluation/clip_ciplab"
"""
train_dataset_list = ["indonesian_deepfake_v2",
                "reswapper_v2_test",
                "inswapper",
                "vh_production_face_deepfake_v1_eval",
                "df40_train_reduced",
                #"df40_eval_fs",
                "facebook_dfdc_val_reduced",
                #"faceforensics++_test",
                "ffhq"]

test_dataset_list = ["df40_eval_fs",#"bing_crawl",
                "faceforensics++_test",
                ]

dict_weights = {'df40_train_reduced': 1,
 'vh_production_face_deepfake_v1_eval': 1,
 'ffhq': 1,
 'inswapper': 5,
 'indonesian_deepfake_v2': 1,
 'reswapper_v2_test': 5,
 'df40_eval_fs': 10,
 'bing_crawl': 20,
 'facebook_dfdc_val_reduced': 10,
 'faceforensics++_test': 1}

dict_folder = {
    "indonesian_deepfake_v2":"private/indonesian_deepfake_dataset_v2",
    "df40_train_reduced":"public/df40_train_reduced",
    "facebook_dfdc_val_reduced":"public/facebook_dfdc_val_reduced",
    "reswapper_v2_test":"private/reswapper_v2_test",
    "inswapper":"private/inswapper",
    "ffhq":"public/ffhq",
    "faceforensics++_test":"public/faceforensics++_test",
    "vh_production_face_deepfake_v1_eval":"private/vh_production_face_deepfake_v1_eval",
    "bing_crawl":"private/bing_crawl",
    #"df40_eval_v2":"public/df40_eval_v2" 
    "df40_eval_fs":"public/df40_eval_fs",
    "client_bcad":"private/client_bcad",
    "client_maybank":"private/client_maybank",
    "client_raya":"private/client_raya",
    "client_TWM":"private/client_TWM",
}
"""

"""
def prepare_training_data(config):
    train_set = StratifiedSourceDataset(
            config['train_dataset'],json_folder=config['dataset_json_folder'], train=True
        )
    live_source_indices_dict = train_set.live_source_indices
    str_live = []
    for k,v in live_source_indices_dict.items():
        str_live.append(f"{k} {len(v)}")

    str_live = '|'.join(str_live)
    logger.info(str_live)

    deepfake_source_indices_dict = train_set.deepfake_source_indices
    str_deepfake = []
    for k,v in deepfake_source_indices_dict.items():
        str_deepfake.append(f"{k} {len(v)}")
    
    str_deepfake = '|'.join(str_deepfake)
    logger.info(str_deepfake)
"""

