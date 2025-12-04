
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


class DeepFakeDataset(Dataset):
    def __init__(self, json_paths,json_folder=None, train=True, ssl=False, transform=None):
        self.data = []
        if isinstance(json_paths, str):
            json_path = json_paths
            if json_folder:
                json_path = os.path.join(json_folder,json_path+'.json')

            with open(json_path, 'r') as f:
                self.data.extend(json.load(f).items())

        else:
            for json_path in json_paths:
                if json_folder:
                    json_path = os.path.join(json_folder,json_path+'.json')

                with open(json_path, 'r') as f:
                    self.data.extend(json.load(f).items())

        self.train = train
        self.ssl = ssl

        if transform is None:
            if not train:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    #T.Resize(384),
                    #T.ToTensor(),
                    # Because prefetcher
                    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                to_tensor = [T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954,0.26130258,0.27577711])]
                
                self.transform = T.Compose(self.transform.transforms + to_tensor)

            else:
                # self.transform = T.Compose([
                #     T.RandomHorizontalFlip(),
                #     T.ToTensor(),
                #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # ])
                self.transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    #T.RandomVerticalFlip(p=0.5),
                    low_quality.RandomDownUpSampler(p=0.25),
                    low_quality.SimulateLowQuality(p=0.25),
                    crop.RandomZoomCrop(perc_range=(0.9, 1.0)),
                    border.RandomBorder(border_amount=(0.001, 0.15), p=0.5),
                    #border.RandomBorder(border_amount=(0.001, 0.15), p=0.5),
                    #T.Resize(448),
                    T.Resize(224),
                    # because prefetcher
                    #T.ToTensor(),
                    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Disable this if using prefetcher
                ])

                to_tensor = [T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954,0.26130258,0.27577711])]

                self.transform = T.Compose(self.transform.transforms + to_tensor)

        else:
            self.transform = transform
        
        self.data_dict = {
            'image': [i[0] for i in self.data],
            'label': [i[1] for i in self.data], 
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, metadata = self.data[idx]
        
        label = metadata['label']  # 0 = real, 1 = deepfake
        img_path = metadata['processed_path']
        #if 'DF40/eval/' in img_path:
        #    img_path = os.path.join("/mnt/ssd2/dataset/"+img_path)


        # Step 1 & 2: Load image with OpenCV and convert to RGB
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        """
        # Step 3: Resize using cv2.resize
        img = cv2.resize(img, (448, 448))  # Resize to 224x224

        # Step 4: Optional manipulations (e.g., for training)
        if self.train:
            # Convert to HSV for saturation adjustment
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if np.random.randint(2):  # Randomly adjust saturation
                img[..., 1] = np.clip(img[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        """
        # Step 5: Convert to PIL Image
        img = Image.fromarray(img)
        # img = Image.fromarray(img).resize((224, 224))

        # Step 6: Apply transforms and return
        if self.train:
            if self.ssl:
                # For self-supervised learning, return multiple augmented views
                return self.transform(img), self.ssl_transforms(img), self.ssl_transforms(img), label
            else:
                return self.transform(img), label
        else:
            #return self.transform(img), label, img_path
            return self.transform(img), label


class StratifiedSourceDataset(Dataset):
    def __init__(self, json_paths, json_folder=None, train=True, ssl=False, 
                 transform=None, dataset_percentage=None):
        
        # Dummy data
        #self.data = torch.arange(100)  # pretend inputs
        self.data = []
        self.live_image_paths = []
        self.live_labels = []
        self.live_sources = []
        self.deepfake_image_paths = []
        self.deepfake_labels = []
        self.deepfake_sources = []
        self.dataset_percentage = dataset_percentage
        for json_path in json_paths:
            if json_folder:
                json_path = os.path.join(json_folder,json_path+'.json')

            with open(json_path, 'r') as f:
                metadata = json.load(f)
            metadata_keys = list(metadata.keys())
            json_name = json_path.split("/")[-1].split(".")[0]
            if dataset_percentage is not None:
                if json_name in dataset_percentage:
                    metadata_keys = metadata_keys[:int(dataset_percentage[json_name]*len(metadata_keys))]
            
            metadata = {d: metadata[d] for d in metadata_keys}
            for img_path, info in metadata.items():
                self.process_data(info, img_path, json_path)
                
            #with open(json_path, 'r') as f:
            #    self.data.extend(json.load(f).items())
        
        self.indices_source_live, self.indices_source_deepfake = self._build_source_indices()

        self.live_len = len(self.live_image_paths)
        self.deepfake_len = len(self.deepfake_image_paths)
        self.train = train
        self.ssl = ssl

        if transform is None:
            if not train:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop((224,224)),
                    #T.Resize(384),
                    #T.CenterCrop((384,384)),
                    #.ToTensor(),
                    # Because prefetcher
                    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                to_tensor = [T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954,0.26130258,0.27577711])]
                
                self.transform = T.Compose(self.transform.transforms + to_tensor)
            else:
                # self.transform = T.Compose([
                #     T.RandomHorizontalFlip(),
                #     T.ToTensor(),
                #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # ])
                self.transform = T.Compose([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    #T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    #T.RandomGrayscale(p=0.2),
                    T.RandomApply([low_quality.SimCLRGaussianBlur(sigma=[.1, 2.])], p=0.5),     
                    low_quality.RandomDownUpSampler(p=0.25,downsampling_range=(0.33,0.8)),
                    low_quality.SimulateLowQuality(p=0.5),
                    
                    #crop.RandomZoomCrop(perc_range=(0.75, 1.0)),
                    #border.RandomBorder(border_amount=(0.001, 0.15), p=0.5),
                    #T.Resize(224),
                    #T.CenterCrop((224,224)),
                    
                    T.RandomResizedCrop(size=(224, 224), scale=(0.66, 1.0)),
                    border.RandomBlackBorderFixedSizeSquare(max_border_ratio=0.3, p=0.5),
                    #T.Resize(224),
                    
                    #T.RandomVerticalFlip(p=0.5),
                    #T.Resize(384),
                    #T.CenterCrop((384,384)),
                    # because prefetcher
                    #T.ToTensor(),
                    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Disable this if using prefetcher
                ])

                to_tensor = [T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954,0.26130258,0.27577711])]

                self.transform = T.Compose(self.transform.transforms + to_tensor)

                if ssl:
                    self.ssl_transforms = T.Compose([
                        T.RandomResizedCrop(224, scale=(0.08, 1.)),
                        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        T.RandomGrayscale(p=0.2),
                        T.RandomApply([low_quality.SimCLRGaussianBlur(sigma=[.1, 2.])], p=0.5),  # Corrected usage
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        else:
            self.transform = transform

        # For Curriculum Learning
        dfd_json_path = "/mnt/ssd/workspace/adi/repos/vh_deepfake_trainer/tools/dataset_assessor/results_att_fd3-1-0/test_200k_merged_train/dfd_1-0-0.json"
        self.live_dfty_scores, self.deepfake_dfty_scores = self.get_difficulty_scores(dfd_json_path)
        self.validate_by_difficulty()
        
        self.data_dict = {
            'image': self.live_image_paths+self.deepfake_image_paths, 
            'label': self.live_labels+self.deepfake_labels, 
            #difficulty_score
        }
        #print(self.transform)
        
    def _build_source_indices(self):
        indices_source_live = defaultdict(list)
        for idx, source in enumerate(self.live_sources):
            indices_source_live[source].append(idx)
        
        indices_source_deepfake = defaultdict(list)
        for idx, source in enumerate(self.deepfake_sources):
            indices_source_deepfake[source].append(idx)
        return dict(indices_source_live), dict(indices_source_deepfake)
    

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
                #live_images.append(self.transform(live_image))
                #deepfake_images.append(self.transform(deepfake_image))
                #live_labels.append(live_label)
                #deepfake_labels.append(deepfake_label)
                #self.live_sources[live_idx] 
                #self.deepfake_sources[deepfake_idx]
                return (self.transform(live_image), self.transform(deepfake_image), 
                        live_label, deepfake_label,
                        self.live_dfty_scores[live_idx], self.deepfake_dfty_scores[deepfake_idx])
            
                #self.live_sources[live_idx], self.deepfake_sources[deepfake_idx] 
                
                
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

    def get_difficulty_scores(self,dfd_json_path):
        path_json = dfd_json_path
        with open(path_json, 'r') as file:
            data_json = json.load(file)

        data_json_new = {v["processed_path"]:v for k,v in data_json.items()}
        live_dfty_scores = [data_json_new[i]['dfd_1-0-0'] if 'dfd_1-0-0' in data_json_new[i] 
                                else -100 for i in self.live_image_paths]
        deepfake_dfty_scores = [1-data_json_new[i]['dfd_1-0-0'] if 'dfd_1-0-0' in data_json_new[i] 
                                    else -100 for i in self.deepfake_image_paths]
        return live_dfty_scores, deepfake_dfty_scores
        self.live_valid_indices = [i for i in range(len(self.live_dfty_scores)) if live_dfty_scores[i] >= 0]
        self.deepfake_valid_indices = [i for i in range(len(self.deepfake_dfty_scores)) if deepfake_dfty_scores[i] >= 0]
    
    def validate_by_difficulty(self):
        live_valid_indices = [i for i in range(len(self.live_dfty_scores)) if self.live_dfty_scores[i] >= 0]
        deepfake_valid_indices = [i for i in range(len(self.deepfake_dfty_scores)) if self.deepfake_dfty_scores[i] >= 0]
        temp = [self.live_image_paths[i] for i in live_valid_indices]
        self.live_image_paths = temp
        temp = [self.live_dfty_scores[i] for i in live_valid_indices]
        self.live_dfty_scores = temp
        temp = [self.deepfake_image_paths[i] for i in deepfake_valid_indices]
        self.deepfake_image_paths = temp
        temp = [self.deepfake_dfty_scores[i] for i in deepfake_valid_indices]
        self.deepfake_dfty_scores = temp

    def progressive_curriculum(self,ds_easy, ds_medium, mix_ratio=0.3):
        # mix 30% of medium difficulty into easy stage
        import random
        idx_easy = list(range(len(ds_easy)))
        idx_medium = random.sample(range(len(ds_medium)), int(len(ds_medium) * mix_ratio))
        mixed_idx = idx_easy + idx_medium
        return torch.utils.data.Subset(ds_easy.dataset, mixed_idx)
    
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


        self.live_dfty_scores = dataset.live_dfty_scores
        self.deepfake_dfty_scores = dataset.deepfake_dfty_scores
    
    def set_active_bins(self, bin_ids):
        self.active_bins = bin_ids

    def make_bins(self,scores, n_bins=3):
        return np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    
    def group_by_source_and_difficulty(self,indices_by_source, dfty_scores, bin_edges):
        grouped = {}

        for src, indices in indices_by_source.items():
            src_groups = {i: [] for i in range(len(bin_edges) - 1)}

            for idx in indices:
                score = dfty_scores[idx]

                # find which bin this score belongs to
                bin_id = np.digitize(score, bin_edges) - 1
                bin_id = max(0, min(bin_id, len(bin_edges) - 2))  # clamp

                src_groups[bin_id].append(idx)

            grouped[src] = src_groups

        return grouped

    def __iter__(self):
        bin_edges_live = self.make_bins(self.live_dfty_scores, n_bins=3)

        live_grouped = self.group_by_source_and_difficulty(
            self.indices_source_live,
            self.live_dfty_scores,
            bin_edges_live
        )

        #pools_source_live = {
        #    src: self.rng.permutation(indices).tolist()
        #    for src, indices in self.indices_source_live.items()
        #}
        pools_source_live = {
            src: {
                bin_id: self.rng.permutation(bin_indices).tolist()
                for bin_id, bin_indices in bins.items() if bin_id in self.active_bins
            }
            for src, bins in live_grouped.items()
        }
        pools_source_live = {
            src: [idx for bin_list in bins.values() for idx in bin_list]
            for src, bins in pools_source_live.items()
        }

        bin_edges_deepfake = self.make_bins(self.deepfake_dfty_scores, n_bins=3)

        deepfake_grouped = self.group_by_source_and_difficulty(
            self.indices_source_deepfake,
            self.deepfake_dfty_scores,
            bin_edges_deepfake
        )

        #pools_source_deepfake = {
        #    src: self.rng.permutation(indices).tolist()
        #    for src, indices in self.indices_source_deepfake.items()
        #}

        pools_source_deepfake = {
            src: {
                bin_id: self.rng.permutation(bin_indices).tolist()
                for bin_id, bin_indices in bins.items() if bin_id in self.active_bins
            }
            for src, bins in deepfake_grouped.items()
        }
        pools_source_deepfake = {
            src: [idx for bin_list in bins.values() for idx in bin_list]
            for src, bins in pools_source_deepfake.items()
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
            
            #shuffle batch
            self.rng.shuffle(batch_live)
            self.rng.shuffle(batch_deepfake)

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

class CurriculumImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples -> (image_path, label, difficulty_score)
        """
        self.samples = sorted(samples, key=lambda x: x[2])  # sort by difficulty
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, difficulty = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, difficulty

    def split_curriculum(dataset, splits=[0.3, 0.3, 0.4]):
        N = len(dataset)
        c1 = int(N * splits[0])
        c2 = int(N * (splits[0] + splits[1]))
        
        ds_easy = torch.utils.data.Subset(dataset, range(0, c1))
        ds_medium = torch.utils.data.Subset(dataset, range(c1, c2))
        ds_hard = torch.utils.data.Subset(dataset, range(c2, N))
        
        return [ds_easy, ds_medium, ds_hard]
    
    def progressive_curriculum(ds_easy, ds_medium, mix_ratio=0.3):
        # mix 30% of medium difficulty into easy stage
        idx_easy = list(range(len(ds_easy)))
        idx_medium = random.sample(range(len(ds_medium)), int(len(ds_medium) * mix_ratio))
        mixed_idx = idx_easy + idx_medium
        return torch.utils.data.Subset(ds_easy.dataset, mixed_idx)