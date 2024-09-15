import os
import tensorflow as tf
import torch
from utils import logger
from utils.tf_logger import TF_Logger
from utils.hparams import HParams
from utils.dataset_utils import load_data
from Harmony_Transformer import *
from utils.mir_eval_modules import chord_content_metric
import argparse
import warnings
import numpy as np
import pandas as pd
import mir_eval
from utils.mir_eval_modules import root_majmin_ccm_score_calculation

idx2chord = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj', 'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min','N', 'X']


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='e')
parser.add_argument('--kfold', type=int, help='6 fold (0,1,2,3,4,5)',default='e')
parser.add_argument('--last_best_epoch', type=int, default=0)

args = parser.parse_args()
experiment_num = str(args.index)
kfold_num = str(args.kfold)
last_best_epoch = args.last_best_epoch


# Result save path
config = HParams.load("run_config.yaml")
asset_path = config.path['asset_path']
ckpt_path = config.path['ckpt_path']
metrics_path = config.path['metrics_path']

ckpt_folder = 'idx_'+experiment_num+'_'+ kfold_num
ckpt_folder_name = 'idx_'+experiment_num+'_%03d'
ckpt_file_name = 'idx_'+experiment_num+'_%03d.cpkt'
tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', 'idx_'+experiment_num+'_'+kfold_num))
metrics_pth_file = 'idx_'+experiment_num+'_'+ kfold_num+'.pth'
metrics_csv_file = 'idx_'+experiment_num+'_'+ kfold_num+'.csv'

logger.info("==== Test: Experiment Number : %d " % args.index)
logger.info("==== Test: Kfold Number: %d " %args.kfold)

# load data

billboard_train_dataset, billboard_valid_dataset = load_data(os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_739_1.npz'))
aam_train_dataset, aam_valid_dataset = load_data(os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz'))
winterreise_train_dataset, winterreise_valid_dataset = load_data(os.path.join('preprocessed_data', 'winterreise', f'winterreise_data_model_input_final_{kfold_num}_48_1.npz'))

# Model

model = Harmony_Transformer()
# Define placeholders
logger.info("build model...")
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, [None, model._n_steps, model._feature_size], name='encoder_inputs') # shape = [batch_size, n_steps, n_inputs]
y = tf.compat.v1.placeholder(tf.int32, [None, model._n_steps], name='chord_labels') # ground_truth, shape = [batch_size, n_steps]
y_cc = tf.compat.v1.placeholder(tf.int32, [None, model._n_steps], name='chord_change_labels') # ground_truth, shape = [batch_size, n_steps]
y_len = tf.compat.v1.placeholder(tf.int32, shape=[None], name="sequence_lengths")
dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
global_step = tf.compat.v1.placeholder(tf.int32, name='global_step')
slope = tf.compat.v1.placeholder(tf.float32, name='slope')
stochastic_tensor = tf.compat.v1.placeholder(tf.bool, name='stochastic_tensor')

encoder_inputs_embedded, chord_change_logits, chord_change_predictions = model.encoder(x, slope, dropout_rate, is_training)
logits, chord_predictions = model.decoder(x, encoder_inputs_embedded, chord_change_predictions, dropout_rate, is_training)

saver = tf.compat.v1.train.Saver()

# Load model
with tf.compat.v1.Session() as sess:
    model_save_path = os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % last_best_epoch, ckpt_file_name % last_best_epoch)
    saver.restore(sess, model_save_path)

    logger.info("Calculating predictions on the validation set...")

    

    score_list_dict1, song_length_list1, average_score_dict1 = root_majmin_ccm_score_calculation('winterreise', winterreise_valid_dataset, sess, chord_predictions, x, y, y_cc, y_len, dropout_rate, is_training, slope, stochastic_tensor)
    score_list_dict2, song_length_list2, average_score_dict2 = root_majmin_ccm_score_calculation('aam', aam_valid_dataset, sess, chord_predictions, x, y, y_cc, y_len, dropout_rate, is_training, slope, stochastic_tensor)
    score_list_dict3, song_length_list3, average_score_dict3 = root_majmin_ccm_score_calculation('billboard', billboard_valid_dataset, sess, chord_predictions, x, y, y_cc, y_len, dropout_rate, is_training, slope, stochastic_tensor)

score_metrics = ['root', 'majmin', 'ccm']

for m in score_metrics:
    message = '==== %s score on winterreise is %.4f' % (m, average_score_dict1[m])
    logger.info(str(message).replace('.',','))

for m in score_metrics:
    message = '==== %s score on aam is %.4f' % (m, average_score_dict2[m])
    logger.info(str(message).replace('.',','))

for m in score_metrics:
    message = '==== %s score on billboard is %.4f' % (m, average_score_dict3[m])
    logger.info(str(message).replace('.',','))

average_score = {}
for m in score_metrics:
    average_score[m] = (np.sum(song_length_list1) * average_score_dict1[m] + np.sum(song_length_list2) * average_score_dict2[m] + np.sum(song_length_list3) * average_score_dict3[m]) / (np.sum(song_length_list1) + np.sum(song_length_list2) + np.sum(song_length_list3))
    message = '==== %s mix average score is %.4f' % (m, average_score[m])
    logger.info(str(message).replace('.',','))


# save full score information
metrics_save_pth = os.path.join(asset_path, metrics_path, ckpt_folder, metrics_pth_file)
state_dict = {'score_list_dict1': score_list_dict1,'song_length_list1': song_length_list1, 'average_score_dict1': average_score_dict1,
            'score_list_dict2': score_list_dict2,'song_length_list2': song_length_list2, 'average_score_dict2': average_score_dict2,
            'score_list_dict3': score_list_dict3,'song_length_list3': song_length_list2, 'average_score_dict3': average_score_dict3,
            'average_score': average_score}
torch.save(state_dict, metrics_save_pth)


# save avergae metrics in csv for convinience
metrics_save_csv = os.path.join(asset_path, metrics_path, ckpt_folder, metrics_csv_file)
aam_df = pd.DataFrame.from_dict(average_score_dict1, 'index')
aam_df = aam_df.rename(columns={0: 'aam'})
winterreise_df = pd.DataFrame.from_dict(average_score_dict2, 'index')
winterreise_df = winterreise_df.rename(columns={0: 'winterreise'})
billboard_df = pd.DataFrame.from_dict(average_score_dict3, 'index')
billboard_df = billboard_df.rename(columns={0: 'billboard'})
average_df = pd.DataFrame.from_dict(average_score, 'index')
average_df = average_df.rename(columns={0: 'average'})
df = pd.concat([aam_df, winterreise_df, billboard_df, average_df], axis=1)
df = df.transpose().round(4)
df.to_csv(metrics_save_csv)

message = '==== copyable metrics: %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (average_score_dict2['root'], average_score_dict2['majmin'], average_score_dict2['ccm'], average_score_dict1['root'], average_score_dict1['majmin'], average_score_dict1['ccm'], average_score_dict3['root'], average_score_dict3['majmin'], average_score_dict3['ccm'])
logger.info(str(message).replace('.',','))