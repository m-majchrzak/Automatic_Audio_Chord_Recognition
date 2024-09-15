import os
from torch import optim
from utils import logger
from audio_dataset import AudioDataset, AudioDataLoader
from utils.tf_logger import TF_Logger
from btc_model import *
from utils.hparams import HParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kfold', type=int, help='6 fold (0,1,2,3,4,5)',default='e')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

config = HParams.load("run_config.yaml")

# Result save path

kfold_num = str(args.kfold)
# Data loader
# train_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=("aam",), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_names=("winterreise",), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)

# train_dataloader1 = AudioDataLoader(dataset=train_dataset1, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
train_dataloader2 = AudioDataLoader(dataset=train_dataset2, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)


# Global mean and variance calculate
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join(config.path['root_path'], 'features', mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')

if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    logger.info("Mean from file: %d" % mean)
    logger.info("Std from file: %d" % std)

mean = 0
square_mean = 0
k = 0
for i, data in enumerate(train_dataloader2):
    features, input_percentages, chords, collapsed_chords, chord_lens, boundaries = data
    features = features.to(device)
    mean += torch.mean(features).item()
    square_mean += torch.mean(features.pow(2)).item()
    k += 1
square_mean = square_mean / k
mean = mean / k
std = np.sqrt(square_mean - mean * mean)
normalization = dict()
normalization['mean'] = mean
normalization['std'] = std
logger.info("Winterreise mean: %d" % mean)
logger.info("Winterreise std: %d" % std)

    
