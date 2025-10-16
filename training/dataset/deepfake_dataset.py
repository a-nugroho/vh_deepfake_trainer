
import torch
from torch.utils.data import Dataset, Subset, Sampler, DataLoader
import json
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
import torchvision.transforms as T

import dataset.utils.aug.low_quality as low_quality
import dataset.utils.aug.crop as crop
import dataset.utils.aug.border as border

from collections import defaultdict



class DeepfakeDataset(Dataset):
    def __init__(self, config,
                train = True, 
                transform = None,
                normalize = True,
                aug_rotate = True,
                aug_blur = True,
                aug_lowq = True,
                aug_crop = True,
                pair_mode = False,
                mean = [0.48145466, 0.4578275, 0.40821073],
                std = [0.26862954,0.26130258,0.27577711],
                size_image = 224,ssl = False):
        
        paths_json = config["paths_json"]
        folder_json = config["folder_json"]
        self.pair_mode = pair_mode
        # Dummy data
        #self.data = torch.arange(100)  # pretend inputs
        transform_default = [T.Resize(size_image),T.CenterCrop((size_image,size_image))]

        self.data = []
        self.paths_image_live = []
        self.paths_image_deepfake = []
        self.labels_live = []
        self.labels_deepfake = []
        self.sources_live = []
        self.sources_deepfake = []

        if isinstance(paths_json, str):
            path_json = paths_json
            if folder_json:
                path_json = os.path.join(folder_json,path_json+'.json')

            with open(path_json, 'r') as f:
                self.data.extend(json.load(f).items())
        else:
            for path_json in paths_json:
                if folder_json:
                    path_json = os.path.join(folder_json,path_json+'.json')

                with open(path_json, 'r') as f:
                    metadata = json.load(f)

                for path_img, info_img in metadata.items():
                    self.process_data(info_img, path_img, path_json)
            
                
        self.indices_source_live, self.indices_source_deepfake = self._build_source_indices()

        self.len_live = len(self.paths_image_live)
        self.len_deepfake = len(self.paths_image_deepfake)

        self.train = train
        self.ssl = ssl

        to_tensor = [T.ToTensor()]
            
        if normalize:
            to_tensor = to_tensor + [T.Normalize(mean=mean,std=std)]
        
        def_hor_flip = 0.5
        def_random_rot = 15
        def_gauss_blur = [.1, 2.]
        def_gauss_p = 0.5
        def_downsampling_range = (0.33,0.8)
        def_downsampling_p = 0.25
        def_simlow_p = 0.5

        transform_aug_rotate = [T.RandomHorizontalFlip(p=def_hor_flip),
                    T.RandomRotation(degrees=def_random_rot)]
        transform_aug_blur = [T.RandomApply([low_quality.SimCLRGaussianBlur(sigma=def_gauss_blur)], p=def_gauss_p)]
        transform_aug_lowq = [low_quality.RandomDownUpSampler(p=def_downsampling_p,downsampling_range=def_downsampling_range),
                    low_quality.SimulateLowQuality(p=def_simlow_p)]
        transform_aug_crop = [crop.RandomZoomCrop(perc_range=(0.9, 1.0)), border.RandomBorder(border_amount=(0.001, 0.15), p=0.5)]

        if transform is None:
            if not train:
                self.transform =  transform_default
            
            else:
                self.transform = []
                if aug_rotate:
                    self.transform.extend(transform_aug_rotate)
                if aug_blur:
                    self.transform.extend(transform_aug_blur)
                if aug_lowq:
                    self.transform.extend(transform_aug_lowq)
                if aug_crop:
                    self.transform.extend(transform_aug_crop)
                
                self.transform.extend(transform_default)

            self.transform = T.Compose(self.transform + to_tensor)
        
        else:
            self.transform = transform

        if self.pair_mode:
            self.data_dict = {
                'image': self.paths_image_live +self.paths_image_deepfake, 
                'label': self.labels_live+self.labels_deepfake, 
            }
        else:
            self.data_dict = {
                'image': [i[0] for i in self.data],
                'label': [i[1] for i in self.data], 
            }
        
    def _build_source_indices(self):
        indices_source_live = defaultdict(list)
        for idx, source in enumerate(self.sources_live):
            indices_source_live[source].append(idx)
        
        indices_source_deepfake = defaultdict(list)
        for idx, source in enumerate(self.sources_deepfake):
            indices_source_deepfake[source].append(idx)
        return dict(indices_source_live), dict(indices_source_deepfake)

    def __len__(self):
        return max(self.len_live, self.len_deepfake)
    
    def __min_len__(self):
        return min(self.len_live, self.len_deepfake)
    
    def load_image(self, path_image):
        try:
            image = Image.open(path_image).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image at path {path_image}: {e}")
            return None
        
    def __getitem__(self, tuple_idxs):
        if self.pair_mode:
            idx_live, idx_deepfake = tuple_idxs
            while True:
                try:
                    # Get live image and its label
                    idx_live = idx_live % self.len_live #Ensure no overflow
                    image_live = self.load_image(
                        self.paths_image_live[idx_live]
                    )
                    live_label = self.labels_live[idx_live]

                    # Get deepfake image and its label
                    idx_deepfake = idx_deepfake % self.len_deepfake
                    image_deepfake = self.load_image(
                        self.paths_image_deepfake[idx_deepfake]
                    )
                    label_deepfake = self.labels_deepfake[idx_deepfake]
                    return self.transform(image_live), self.transform(image_deepfake), live_label, label_deepfake
                    #self.sources_live[live_idx], self.sources_deepfake[deepfake_idx] 
                    
                    
                except Exception as e:
                    print(f"Error loading image at index {idx_live}: {e}")
                    idx_live = (idx_live + 1) % self.len_live
        
        else:
            idx = tuple_idxs
            path_image, metadata = self.data[idx]
        
            label = metadata['label']
            path_image = metadata['processed_path']
            image = self.load_image(path_image)
            image = Image.fromarray(image)
            
            return self.transform(image), label

    def process_data(self, info, path_json):
        """
        Processes training data and appends paths and labels for both live and deepfake images.

        Args:
            info (dict): Metadata of the image.
            path_image (str): Path to the image in the JSON metadata.
        """
        if self.pair_mode:
            if info["label"] == 0:
                self.paths_image_live.append(
                    os.path.join("data", info["processed_path"])
                )
                self.labels_live.append(info["label"])
                self.sources_live.append(path_json)

            else:
                self.paths_image_deepfake.append(
                    os.path.join("data", info["processed_path"])
                )
                
                self.labels_deepfake.append(info["label"])
                self.sources_deepfake.append(path_json)

        else:
            with open(path_json, 'r') as f:
                self.data.extend(json.load(f).items())


class ProportionalStratifiedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, seed=42, print_info=True):
        #assert hasattr(dataset, 'source_to_indices'), "Dataset must have 'source_to_indices'"
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.indices_source_live = dataset.indices_source_live
        self.indices_source_deepfake = dataset.indices_source_deepfake

        str_live = []
        for k,v in self.indices_source_live.items():
            str_live.append(f"{k} {len(v)}")

        str_live = '|'.join(str_live)
        
        str_deepfake = []
        for k,v in self.indices_source_deepfake.items():
            str_deepfake.append(f"{k} {len(v)}")
        
        str_deepfake = '|'.join(str_deepfake)
        if print_info:
            print(str_live)
            print(str_deepfake)
            
        self.sources_live = list(self.indices_source_live.keys())
        self.sources_deepfake = list(self.indices_source_deepfake.keys())
        total_samples = len(dataset)
        
        self.batch_sizes_source_live = {
            src: int(np.floor(batch_size/len(self.sources_live)))
            for src, indices in self.indices_source_live.items()
        }
        
        self.batch_sizes_source_deepfake = {
            src: int(np.floor(batch_size/len(self.sources_deepfake)))
            for src, indices in self.indices_source_deepfake.items()
        }
        
        # Adjust to make sum exactly batch_size
        i = 0 
        while sum(self.batch_sizes_source_live.values()) < batch_size:
            self.batch_sizes_source_live[sorted(list(self.indices_source_live.keys()))[i]] += 1
            i+=1
            
        i = 0 
        while sum(self.batch_sizes_source_deepfake.values()) < batch_size:
            self.batch_sizes_source_deepfake[sorted(list(self.indices_source_deepfake.keys()))[i]] += 1
            i+=1

        if print_info:
            print(self.batch_sizes_source_live)
            print(self.batch_sizes_source_deepfake)
            
    def __iter__(self):
        pools_source_live = {
            src: self.rng.permutation(indices).tolist()
            for src, indices in self.indices_source_live.items()
        }

        pools_source_deepfake = {
            src: self.rng.permutation(indices).tolist()
            for src, indices in self.indices_source_deepfake.items()
        }

        finished = False
        total_batch = len(self)
        iter_batch = 0
        
        while iter_batch < total_batch:
            batch_live = []
            for src in self.sources_live:
                pool = pools_source_live[src]
                needed = self.batch_sizes_source_live[src]

                if len(pool) < needed:
                    if self.drop_last:
                        finished = True
                        break
                    else:
                        pool += self.rng.permutation(self.indices_source_live[src]).tolist()

                batch_live.extend(pool[:needed])
                pools_source_live[src] = pool[needed:]

            batch_deepfake = []
            for src in self.sources_deepfake:
                pool = pools_source_deepfake[src]
                needed = self.batch_sizes_source_deepfake[src]

                if len(pool) < needed:
                    if self.drop_last:
                        finished = True
                        break
                    else:
                        pool += self.rng.permutation(self.indices_source_deepfake[src]).tolist()

                batch_deepfake.extend(pool[:needed])
                pools_source_deepfake[src] = pool[needed:]

            if finished or (self.drop_last and len(batch_live) < self.batch_size):
                break
            
            self.rng.shuffle(batch_live)
            self.rng.shuffle(batch_deepfake)
            #shuffle batch
            yield zip(batch_live,batch_deepfake)
            iter_batch += 1

    def __len__(self):
        
        ab = max(
            len(self.indices_source_live[src]) // max(1, self.batch_sizes_source_live[src])
            for src in self.sources_live
        )
        cd = max(
            len(self.indices_source_deepfake[src]) // max(1, self.batch_sizes_source_deepfake[src])
            for src in self.sources_deepfake
        )
        return max(ab,cd) 
