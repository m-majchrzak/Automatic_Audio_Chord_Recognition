import os
import tensorflow as tf
from utils import logger
from utils.tf_logger import TF_Logger
from utils.hparams import HParams
from utils.dataset_utils import load_data, merge_data
from Harmony_Transformer import *
import argparse
import warnings
import numpy as np
from utils.mir_eval_modules import root_majmin_ccm_score_calculation
import torch
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logger.logging_verbosity(1)
tf.compat.v1.disable_eager_execution()

# Setup 
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, help='Experiment Number', default='e')
parser.add_argument('--kfold', type=int, help='6 fold (0,1,2,3,4,5)',default='e')
parser.add_argument('--dataset', type=str, help='Whole dataset label, training and valid', default='aam_winterreise')
parser.add_argument('--early_stop', type=bool, help='no improvement during 10 epoch -> stop', default=1)
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
ckpt_folder_name = 'idx_'+experiment_num+'_%03d'
ckpt_file_name = 'idx_'+experiment_num+'_%03d.cpkt'
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

## Load data
logger.info("Load input data...")

billboard_train_dataset, billboard_valid_dataset = load_data(os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_739_1.npz'))
aam_train_dataset, aam_valid_dataset = load_data(os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz'))
winterreise_train_dataset, winterreise_valid_dataset = load_data(os.path.join('preprocessed_data', 'winterreise', f'winterreise_data_model_input_final_{kfold_num}_48_1.npz'))

if args.dataset == "billboard":
    train_dataset = billboard_train_dataset
    valid_dataset = billboard_valid_dataset 

elif args.dataset == "aam":
    train_dataset = aam_train_dataset
    valid_dataset = aam_valid_dataset 

elif args.dataset == "winterreise":
    train_dataset = winterreise_train_dataset
    valid_dataset = winterreise_valid_dataset 

elif args.dataset == "billboard_aam":
    billboard_file_dir = os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_192_1.npz')
    billboard_train_dataset, billboard_valid_dataset = load_data(billboard_file_dir)
    aam_file_dir = os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz')
    aam_train_dataset, aam_valid_dataset = load_data(aam_file_dir)
    train_dataset, valid_dataset = merge_data([billboard_file_dir, aam_file_dir])

elif args.dataset == "billboard_winterreise":
    billboard_file_dir = os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_192_1.npz')
    billboard_dataset = load_data(billboard_file_dir)
    winterreise_file_dir = os.path.join('preprocessed_data', 'winterreise', f'winterreise_data_model_input_final_{kfold_num}_48_4.npz')
    winterreise_dataset = load_data(winterreise_file_dir)
    train_dataset, valid_dataset = merge_data([billboard_file_dir, winterreise_file_dir])

elif args.dataset == "aam_winterreise":
    aam_file_dir = os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz')
    aam_dataset = load_data(aam_file_dir)
    winterreise_file_dir = os.path.join('preprocessed_data', 'winterreise', f'winterreise_data_model_input_final_{kfold_num}_48_4.npz')
    winterreise_dataset = load_data(winterreise_file_dir)
    train_dataset, valid_dataset = merge_data([aam_file_dir, winterreise_file_dir])

elif args.dataset == "billboard_aam_winterreise":
    billboard_file_dir = os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_192_1.npz')
    billboard_dataset = load_data(billboard_file_dir)
    aam_file_dir = os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz')
    aam_dataset = load_data(aam_file_dir)
    winterreise_file_dir = os.path.join('preprocessed_data', 'winterreise', f'winterreise_data_model_input_final_{kfold_num}_48_4.npz')
    winterreise_dataset = load_data(winterreise_file_dir)
    train_dataset, valid_dataset = merge_data([billboard_file_dir, aam_file_dir, winterreise_file_dir])
else:
    logger.error(f"Unknown dataset name:{args.dataset}")


x_train = train_dataset['x']
TC_train = train_dataset['TC']
y_train = train_dataset['y']
y_cc_train = train_dataset['y_cc']
y_len_train = train_dataset['y_len']
x_valid = valid_dataset['x']
TC_valid = valid_dataset['TC']
y_valid = valid_dataset['y']
y_cc_valid = valid_dataset['y_cc']
y_len_valid = valid_dataset['y_len']

num_examples_train = train_dataset['x'].shape[0]
num_examples_valid = valid_dataset['x'].shape[0]

# print(f'x_train: {np.shape(train_dataset['x'])}')
# print(f'TC_train: {np.shape(train_dataset['TC'])}')
# print(f'y_train: {np.shape(train_dataset['y'])}')
# print(f'y_cc_train: {np.shape(train_dataset['y_cc'])}')
# print(f'y_len_train: {np.shape(train_dataset['y_len'])}')
# print(f'x_valid: {np.shape(valid_dataset['x'])}')
# print(f'TC_valid: {np.shape(valid_dataset['TC'])}')
# print(f'y_valid: {np.shape(valid_dataset['y'])}')
# print(f'y_cc_valid: {np.shape(valid_dataset['y_cc'])}')
# print(f'y_len_valid: {np.shape(valid_dataset['y_len'])}')


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

# Define loss
with tf.compat.v1.name_scope('loss'):
    loss_ct = model._lambda_loss_ct * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_cc, tf.float32), logits=chord_change_logits))
    loss_c = model._lambda_loss_c * tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y, depth=model._n_classes), logits=logits, label_smoothing=0.1)

    # L2 norm regularization
    vars = tf.compat.v1.trainable_variables()
    L2_regularizer = model._lambda_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

    # loss
    loss = loss_ct + loss_c + L2_regularizer

with tf.compat.v1.name_scope('optimization'):
    # apply learning rate decay
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=model._initial_learning_rate,
                                                global_step=global_step,
                                                decay_steps=(train_dataset['x'].shape[0] // model._batch_size),
                                                decay_rate=0.96,
                                                staircase=True)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.98,
                                        epsilon=1e-9)

    # Apply gradient clipping
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

# Define accuracy
with tf.compat.v1.name_scope('accuracy'):
    label_mask = tf.less(y, 24)  # where label != 24('X)' and label != 25('pad')
    correct_predictions = tf.equal(chord_predictions, y)
    correct_predictions_mask = tf.boolean_mask(tensor=correct_predictions, mask=label_mask)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions_mask, tf.float32))

# Training
logger.info('train the model...')

best_acc = 0
early_stop_idx = 0

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    if restore_epoch > 0:
        if os.path.isfile(os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % restore_epoch, ckpt_file_name % restore_epoch)):
            saver = tf.compat.v1.train.Saver()
            model_restore_path = os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % restore_epoch, ckpt_file_name % restore_epoch)
            saver.restore(sess, model_restore_path)
            logger.info(f"Model restored from checkpoint: {model_restore_path}")

            # Set the global step to the restore_epoch (continuing training from there)
            start_step = restore_epoch * (num_examples_train // model._batch_size)
        else:
            start_step = 0
            logger.info("no checkpoint with %d epochs" % restore_epoch)
    else:
        start_step = 0

    epoch = num_examples_train // model._batch_size # steps per epoch
    annealing_slope = 1.0

    train_loss_list = []
    train_loss_ct_list = []
    train_loss_c_list = []
    train_loss_l2_list = []
    train_acc_list = []

    for step in range(start_step, model._training_steps):

        epoch_number = step // epoch 
        if step % epoch == 0:
            # shuffle trianing set
            indices = random.sample(range(num_examples_train), num_examples_train)
            batch_indices = [indices[x:x + model._batch_size] for x in range(0, len(indices), model._batch_size)]

            if step != start_step:
                annealing_slope *= model._annealing_rate

            train_loss_list = []
            train_loss_ct_list = []
            train_loss_c_list = []
            train_loss_l2_list = []
            train_acc_list = []

        print(f'batch_indices len: {len(batch_indices)}')
        print(f'step: {step}')
        print(f'step % len(batch_indices): {step % len(batch_indices)}')
        # training
        batch = (train_dataset['x'][batch_indices[step % len(batch_indices)]],
                    train_dataset['y_cc'][batch_indices[step % len(batch_indices)]],
                    train_dataset['y'][batch_indices[step % len(batch_indices)]],
                    train_dataset['y_len'][batch_indices[step % len(batch_indices)]],
                    train_dataset['TC'][batch_indices[step % len(batch_indices)]])

        x_batch = batch[0]
        train_run_list = [train_op, loss, loss_ct, loss_c, L2_regularizer, chord_change_predictions, chord_predictions, accuracy]
        train_feed_fict = {x: x_batch,
                            y_cc: batch[1],
                            y: batch[2],
                            y_len: batch[3],
                            dropout_rate: model._dropout_rate,
                            is_training: True,
                            global_step: step + 1,
                            slope: annealing_slope,
                            stochastic_tensor: True}
        _, train_loss, train_loss_ct, train_loss_c, train_L2,  train_cc_pred, train_c_pred, train_acc = sess.run(train_run_list, feed_dict=train_feed_fict)
        
        train_loss_list.append(train_loss)
        train_loss_ct_list.append(train_loss_ct)
        train_loss_c_list.append(train_loss_c)
        train_loss_l2_list.append(train_L2)
        train_acc_list.append(train_acc)

        # logging accuracy
        if step % epoch == 0:

            avg_train_loss = np.mean(train_loss_list)
            avg_train_loss_ct = np.mean(train_loss_ct_list)
            avg_train_loss_c = np.mean(train_loss_c_list)
            avg_train_loss_l2 = np.mean(train_loss_l2_list)
            avg_train_acc = np.mean(train_acc_list)

            result = {'loss/tr': avg_train_loss, 'acc/tr': avg_train_acc}
            for tag, value in result.items(): tf_logger.scalar_summary(tag, value, epoch_number+1)


            logger.info("------ epoch %d, ------ step %d: train_loss %.4f (ct %.4f, c %.4f, L2 %.4f), train_accuracy %.4f ------" % (epoch_number+1, step+1, avg_train_loss, avg_train_loss_ct, avg_train_loss_c, avg_train_loss_l2, avg_train_acc))
            # logging loss and accuracy using tensorboard
            

            # ToDO: validation
            valid_loss_list = []
            valid_acc_list = []
            for i in range(0, num_examples_valid, model._batch_size):
                x_valid_batch = valid_dataset['x'][i:i+model._batch_size]
                y_valid_batch = valid_dataset['y'][i:i+model._batch_size]
                y_cc_valid_batch = valid_dataset['y_cc'][i:i+model._batch_size]
                y_len_valid_batch = valid_dataset['y_len'][i:i+model._batch_size]
                
                val_feed_dict = {x: x_valid_batch,
                                y_cc: y_cc_valid_batch,
                                y: y_valid_batch,
                                y_len: y_len_valid_batch,
                                dropout_rate: 0.0,  # No dropout during evaluation
                                is_training: False,
                                slope: annealing_slope,
                                stochastic_tensor: False}

                val_loss, val_acc = sess.run([loss, accuracy], feed_dict=val_feed_dict)
                valid_loss_list.append(val_loss)
                valid_acc_list.append(val_acc)

            avg_valid_loss = np.mean(valid_loss_list)
            avg_valid_acc = np.mean(valid_acc_list)


            result = {'loss/val': avg_valid_loss, 'acc/val': avg_valid_acc}
            for tag, value in result.items(): tf_logger.scalar_summary(tag, value, epoch_number + 1)

            logger.info("------ epoch %d, ------ step %d: valid_loss %.4f, valid_accuracy %.4f ------" % (epoch_number+1, step+1, avg_valid_loss, avg_valid_acc))

            # save model
            current_acc = avg_valid_acc
            if best_acc < avg_valid_acc:
                early_stop_idx = 0
                best_acc = avg_valid_acc
                logger.info('==== best accuracy is %.4f and epoch is %d' % (best_acc, epoch_number + 1))
                logger.info('saving model, Epoch %d, step %d' % (epoch_number + 1, step + 1))

                # save tensorflow model (weights and optimizer state)
                saver = tf.compat.v1.train.Saver()
                os.makedirs(os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % (epoch_number + 1)))
                model_save_path = os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % (epoch_number + 1), ckpt_file_name % (epoch_number + 1))
                save_path = saver.save(sess, model_save_path)

                last_best_epoch = epoch_number + 1

            elif (epoch_number + 1) % config.experiment['save_step'] == 0:
                logger.info('saving model, Epoch %d, step %d' % (epoch_number + 1, step + 1))
                # save tensorflow model (weights and optimizer state)
                saver = tf.compat.v1.train.Saver()
                os.makedirs(os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % (epoch_number + 1)))
                model_save_path = os.path.join(asset_path, ckpt_path, ckpt_folder, ckpt_folder_name % (epoch_number + 1), ckpt_file_name % (epoch_number + 1))
                save_path = saver.save(sess, model_save_path)

                early_stop_idx += 1
            else:
                early_stop_idx += 1

        if (args.early_stop == True) and (early_stop_idx > 9):
            logger.info('==== early stopped and epoch is %d' % (epoch_number + 1))
            break

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
winterreise_df = pd.DataFrame.from_dict(average_score_dict1, 'index')
winterreise_df = winterreise_df.rename(columns={0: 'winterreise'})
aam_df = pd.DataFrame.from_dict(average_score_dict2, 'index')
aam_df = aam_df.rename(columns={0: 'aam'})
billboard_df = pd.DataFrame.from_dict(average_score_dict3, 'index')
billboard_df = billboard_df.rename(columns={0: 'billboard'})
average_df = pd.DataFrame.from_dict(average_score, 'index')
average_df = average_df.rename(columns={0: 'average'})
df = pd.concat([aam_df, winterreise_df, billboard_df, average_df], axis=1)
df = df.transpose().round(4)
df.to_csv(metrics_save_csv)

message = '==== copyable metrics: %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (average_score_dict2['root'], average_score_dict2['majmin'], average_score_dict2['ccm'], average_score_dict1['root'], average_score_dict1['majmin'], average_score_dict1['ccm'], average_score_dict3['root'], average_score_dict3['majmin'], average_score_dict3['ccm'])
logger.info(str(message).replace('.',','))