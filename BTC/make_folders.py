import os
from torch import optim
import pandas as pd
from utils.hparams import HParams

exp_ids = range(1,14)
kfolds = range(6)

for exp_id in exp_ids:
    for kfold in kfolds:
        experiment_num = str(exp_id)
        kfold_num = str(kfold)

        # Result save path
        config = HParams.load("run_config.yaml")
        asset_path = config.path['asset_path']
        ckpt_path = config.path['ckpt_path']
        metrics_path = config.path['metrics_path']

        ckpt_folder = 'idx_'+experiment_num+'_'+ kfold_num
        ckpt_file_name = 'idx_'+experiment_num+'_%03d.pth.tar'

        metrics_pth_file = 'idx_'+experiment_num+'_'+ kfold_num+'.pth'
        metrics_csv_file = 'idx_'+experiment_num+'_'+ kfold_num+'.csv'

        # Make asset directory
        if not os.path.exists(os.path.join(asset_path, ckpt_path, ckpt_folder)):
            os.makedirs(os.path.join(asset_path, ckpt_path, ckpt_folder))

        # Make metrics directory
        if not os.path.exists(os.path.join(asset_path, metrics_path, ckpt_folder)):
            os.makedirs(os.path.join(asset_path, metrics_path, ckpt_folder))
