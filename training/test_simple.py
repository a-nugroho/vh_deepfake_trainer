# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from collections import defaultdict
#from optimizor.SAM import SAM
#from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
#from training.dataset.deepfake_dataset import DeepFakeDataset
from metrics.utils import get_test_metrics

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+",required=False)
parser.add_argument('--weights_path', type=str)
parser.add_argument("--ddp", action='store_true', default=False)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        test_set = DeepFakeDataset(
            test_name,json_folder=config['dataset_json_folder'], train=False
        )
            
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                #shuffle=True,
                num_workers=int(config['workers']),
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config['optimizer'][opt_name]['lr'],
        weight_decay=config['optimizer'][opt_name]['weight_decay'],
        betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
        eps=config['optimizer'][opt_name]['eps'],
        amsgrad=config['optimizer'][opt_name]['amsgrad'],
    )
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def main():
    global logger
    # parse options and load config
    # base config following detector config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    # update with train_config
    with open('config/vhubs_train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['ddp']= args.ddp
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    
    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    weights_path = args.weights_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # start testing
    # define test recorder
    losses_all_datasets = {}
    metrics_all_datasets = {}
    best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each 
    trainer.logger.info("===> Test start!")
    avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        # save the testing data_dict
        data_dict = test_data_loaders[key].dataset.data_dict
        trainer.save_data_dict('test_direct', data_dict, key)
        
        # compute loss for each dataset
        losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = trainer.test_one_dataset_verihubs(test_data_loaders[key])
        losses_all_datasets[key] = losses_one_dataset_recorder
        metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'],direct_auc=True)
        print(metric_one_dataset)
        for metric_name, value in metric_one_dataset.items():
            if metric_name in avg_metric:
                avg_metric[metric_name]+=value
        avg_metric['dataset_dict'][key] = metric_one_dataset[trainer.metric_scoring]

if __name__ == '__main__':
    main()
