import numpy as np
import random
from os.path import join

from utils import prepare_dirs, save_config
from config import get_config
from trainer import Trainer

import torch


def main(config):
    # Set a random seed for reproducible results
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # ensure directories are setup
    trial_dir = join(config.logs_folder, config.dataset, 'trial'+str(config.trial_num))
    prepare_dirs(trial_dir, config.flush)
        
    if config.is_train:
        try:
            save_config(trial_dir, config)
        except ValueError:
            print(
                "[!] file already exist. Either change the trial number,",
                "or delete the json file and rerun.",
                sep=' ',
            )
    
    trainer = Trainer(trial_dir, config)

    if config.is_train: 
        trainer.train()
    else:
        trainer.test()

            
if __name__ == '__main__':    
    config, unparsed = get_config()
    main(config)
    