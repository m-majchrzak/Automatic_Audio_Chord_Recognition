import os
from torch import optim
from utils import logger
from utils.tf_logger import TF_Logger
from btc_model import *
from utils.hparams import HParams
import argparse
from utils.pytorch_utils import adjusting_learning_rate
from utils.mir_eval_modules import *
import warnings
from utils.mir_eval_modules import metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, help='Ground truth lab directory', default='e')
parser.add_argument('--est_dir', type=str, help='Prediction lab directory', default='e')
parser.add_argument('--audio_dir', type=str, help='Audio directory', default='e')
parser.add_argument('--voca', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--audio_type', type=str, help='flac or wav', default='flac')
args = parser.parse_args()

config = HParams.load("run_config.yaml")

# score Validation
score_metrics = ['root', 'majmin', 'ccm']

metrics_ = metrics()
song_length_list = list()
    
for est_file in os.listdir(args.est_dir):

    song_name = est_file.replace(".lab", "")
    gt_file_path = os.path.join(args.gt_dir, est_file)
    est_file_path = os.path.join(args.est_dir, est_file)

    audio_filename = song_name+'.'+args.audio_type
    audio_file_path = os.path.join(args.audio_dir, audio_filename)
    feature, feature_per_second, song_length_second = audio_file_to_features(audio_file_path, config)
    
    metrics_.score(gt_path=gt_file_path, est_path=est_file_path)
    song_length_list.append(song_length_second)

tmp = song_length_list / np.sum(song_length_list)
for m in score_metrics:
    metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

for m in score_metrics:
    #average_score = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) *average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
    logger.info('==== %s score 1 is %.4f' % (m, metrics_.average_score[m]))
        
        



    
