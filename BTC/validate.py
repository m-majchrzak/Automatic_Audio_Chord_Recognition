import os
from torch import optim
from utils import logger
from audio_dataset import AudioDataset, AudioDataLoader
from utils.tf_logger import TF_Logger
from btc_model import *
from utils.hparams import HParams
import argparse
from utils.pytorch_utils import adjusting_learning_rate
from utils.mir_eval_modules import root_majmin_ccm_score_calculation
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Setup 
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='e')
parser.add_argument('--kfold', type=int, help='6 fold (0,1,2,3,4,5)',default='e')
parser.add_argument('--dataset', type=str, help='Whole dataset label, training and valid', default='aam_winterreise')
parser.add_argument('--dataset1', type=str, help='Dataset', default='aam')
parser.add_argument('--dataset1_subset', type=str, help='Number of song from dataset1', default=3000)
parser.add_argument('--dataset1_multiplier', type=str, help='How many times to multiply songs in for dataset1', default=1)
parser.add_argument('--dataset2', type=str, help='Dataset', default='winterreise')
parser.add_argument('--dataset2_subset', type=str, help='Number of song from dataset2', default=48)
parser.add_argument('--dataset2_multiplier', type=str, help='How many times to multiply songs for dataset2', default=1)
parser.add_argument('--use_pretrained', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--restore_epoch', type=int, default=0)
args = parser.parse_args()

restore_epoch = args.restore_epoch
experiment_num = str(args.index)
kfold_num = str(args.kfold)

# Result save path
config = HParams.load("run_config.yaml")
asset_path = config.path['asset_path']
ckpt_path = config.path['ckpt_path']
metrics_path = config.path['metrics_path']

ckpt_folder = 'idx_'+experiment_num+'_'+ kfold_num
ckpt_file_name = 'idx_'+experiment_num+'_%03d.pth.tar'
tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', 'idx_'+experiment_num+'_'+kfold_num))
metrics_pth_file = 'idx_'+experiment_num+'_'+ kfold_num+'.pth'
metrics_csv_file = 'idx_'+experiment_num+'_'+ kfold_num+'.csv'

# Make asset directory
if not os.path.exists(os.path.join(asset_path, ckpt_path, ckpt_folder)):
    os.makedirs(os.path.join(asset_path, ckpt_path, ckpt_folder))

# Make metrics directory
if not os.path.exists(os.path.join(asset_path, metrics_path, ckpt_folder)):
    os.makedirs(os.path.join(asset_path, metrics_path, ckpt_folder))

logger.info("==== Experiment Number : %d " % args.index)
logger.info("==== Kfold Number: %d " %args.kfold)

# Data loader
train_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_name=args.dataset1, subset=int(args.dataset1_subset), multiplier=int(args.dataset1_multiplier), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)
train_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_name=args.dataset2, subset=int(args.dataset2_subset), multiplier=int(args.dataset2_multiplier), num_workers=20, preprocessing=False, train=True, kfold=args.kfold)

valid_dataset1 = AudioDataset(config, root_dir=config.path['root_path'], dataset_name=args.dataset1, subset=int(args.dataset1_subset), multiplier=int(args.dataset1_multiplier), preprocessing=False, train=False, kfold=args.kfold)
valid_dataset2 = AudioDataset(config, root_dir=config.path['root_path'], dataset_name=args.dataset2, subset=int(args.dataset2_subset), multiplier=int(args.dataset2_multiplier), preprocessing=False, train=False, kfold=args.kfold)

if args.dataset == "aam":
    train_dataset = train_dataset1
    valid_dataset = valid_dataset1
elif args.dataset == "winterreise":
    train_dataset = train_dataset2
    valid_dataset = valid_dataset2
elif args.dataset == "aam_winterreise":
    train_dataset = train_dataset1.__add__(train_dataset2)
    valid_dataset = valid_dataset1.__add__(valid_dataset2)
else:
    logger.error(f"Unknown dataset name:{args.dataset}")


train_dataloader = AudioDataLoader(dataset=train_dataset, batch_size=config.experiment['batch_size'], drop_last=False, shuffle=True)
valid_dataloader = AudioDataLoader(dataset=valid_dataset, batch_size=config.experiment['batch_size'], drop_last=False)

# Model and Optimizer
model = BTC_model(config=config.model).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.experiment['learning_rate'], weight_decay=config.experiment['weight_decay'], betas=(0.9, 0.98), eps=1e-9)

# Load model from file/pretrained
if os.path.isfile(os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_file_name % restore_epoch)):
    checkpoint = torch.load(os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_file_name % restore_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    logger.info("restore model with %d epochs" % restore_epoch)

elif args.use_pretrained is True:

    model_file = './model/btc_model.pt'
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model'])

    logger.info("no checkpoint with %d epochs, loading pretrained model" % restore_epoch)

    # only change last layer
    for name, param in model.named_parameters():
        if 'output_projection' not in name:
            param.requires_grad = False

else:
    logger.info("no checkpoint with %d epochs, loading initial model" % restore_epoch)
    restore_epoch = 0

# Global mean and variance calculate
mp3_config = config.mp3
feature_config = config.feature
mp3_string = "%d_%.1f_%.1f" % (mp3_config['song_hz'], mp3_config['inst_len'], mp3_config['skip_interval'])
feature_string = "_%s_%d_%d_%d_" % ('cqt', feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])
z_path = os.path.join('features', args.dataset , mp3_string + feature_string + 'mix_kfold_'+ str(args.kfold) +'_normalization.pt')

if os.path.exists(z_path):
    normalization = torch.load(z_path)
    mean = normalization['mean']
    std = normalization['std']
    logger.info("Global mean and std (k fold index %d) load complete" % args.kfold)

else:
    mean = 0
    square_mean = 0
    k = 0
    for i, data in enumerate(train_dataloader):
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
    torch.save(normalization, z_path)
    logger.info("Global mean and std (training set, k fold index %d) calculation complete" % args.kfold)


score_metrics = ['root', 'majmin', 'ccm']
score_list_dict1, song_length_list1, average_score_dict1 = root_majmin_ccm_score_calculation(valid_dataset=valid_dataset1, config=config, model=model, mean=mean, std=std, device=device)
score_list_dict2, song_length_list2, average_score_dict2 = root_majmin_ccm_score_calculation(valid_dataset=valid_dataset2, config=config, model=model, mean=mean, std=std, device=device)

for m in score_metrics:
    message = '==== %s score on dataset 1 is %.4f' % (m, average_score_dict1[m])
    logger.info(str(message).replace('.',','))

for m in score_metrics:
    message = '==== %s score on dataset 2 is %.4f' % (m, average_score_dict2[m])
    logger.info(str(message).replace('.',','))

average_score = {}
for m in score_metrics:
    average_score[m] = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2))
    message = '==== %s mix average score is %.4f' % (m, average_score[m])
    logger.info(str(message).replace('.',','))


# save full score information
metrics_save_pth = os.path.join(asset_path, metrics_path, ckpt_folder, metrics_pth_file)
state_dict = {'score_list_dict1': score_list_dict1,'song_length_list1': song_length_list1, 'average_score_dict1': average_score_dict1,
              'score_list_dict2': score_list_dict2,'song_length_list2': song_length_list2, 'average_score_dict2': average_score_dict2,
              'average_score': average_score}
torch.save(state_dict, metrics_save_pth)


# save avergae metrics in csv for convinience
metrics_save_csv = os.path.join(asset_path, metrics_path, ckpt_folder, metrics_csv_file)
aam_df = pd.DataFrame.from_dict(average_score_dict1, 'index')
aam_df = aam_df.rename(columns={0: 'aam'})
winterreise_df = pd.DataFrame.from_dict(average_score_dict2, 'index')
winterreise_df = winterreise_df.rename(columns={0: 'winterreise'})
aam_winterreise_df = pd.DataFrame.from_dict(average_score, 'index')
aam_winterreise_df = aam_winterreise_df.rename(columns={0: 'aam_winterreise'})
df = pd.concat([aam_df, winterreise_df, aam_winterreise_df], axis=1)
df = df.transpose().round(4)
df.to_csv(metrics_save_csv)


