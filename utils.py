import os, json, shutil, types
import numpy as np
from os.path import join, exists
import nibabel as nib
import pandas as pd



class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prepare_dirs(trial_dir, flush):
    if not exists(trial_dir):
        os.makedirs(trial_dir)
    elif flush:
        shutil.rmtree(trial_dir)
        if not exists(trial_dir):
            os.makedirs(trial_dir)


def save_config(trial_dir, config):
    config_path = join(trial_dir, f'config_r{config.repeat_num}s{config.split_num}.json')

    all_params = config.__dict__
    with open(config_path, 'w') as fp:
        json.dump(all_params, fp, indent=4, sort_keys=False)


    