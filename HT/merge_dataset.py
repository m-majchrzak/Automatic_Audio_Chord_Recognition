from utils.dataset_utils import load_data, merge_data
import os
import numpy as np
import pickle

kfold_num=0
billboard_file_dir = os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_192_1.npz')
aam_file_dir = os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz')

# x_train_billboard, TC_train_billboard, y_train_billboard, y_cc_train_billboard, y_len_train_billboard, x_valid_billboard, TC_valid_billboard, y_valid_billboard, y_cc_valid_billboard, y_len_valid_billboard, split_sets_billboard = load_data(billboard_file_dir)    
# x_train_aam, TC_train_aam, y_train_aam, y_cc_train_aam, y_len_train_aam, x_valid_aam, TC_valid_aam, y_valid_aam, y_cc_valid_aam, y_len_valid_aam, split_sets_aam = load_data(aam_file_dir)

billboard_file_dir = os.path.join('preprocessed_data', 'billboard', f'billboard_data_model_input_final_{kfold_num}_192_1.npz')
billboard_train_dataset, billboard_valid_dataset = load_data(billboard_file_dir)
aam_file_dir = os.path.join('preprocessed_data', 'aam', f'aam_data_model_input_final_{kfold_num}_192_1.npz')
aam_train_dataset, aam_valid_dataset = load_data(aam_file_dir)

print(np.shape(billboard_train_dataset['x']))
print(np.shape(aam_train_dataset['x']))
# x_train = np.concatenate((x_train_billboard, x_train_aam), axis=0)

x_train = np.concatenate((billboard_train_dataset['x'], aam_train_dataset['x']), axis=0)
print(np.shape(x_train))

# TC_train = np.concatenate((TC_train, input_data['TC_train']), axis=0)
# y_train = np.concatenate((y_train, input_data['y_train']), axis=0)
# y_cc_train = np.concatenate((y_cc_train, input_data['y_cc_train']), axis=0)
# y_len_train = np.concatenate((y_len_train, input_data['y_len_train']), axis=0)
# x_valid = np.concatenate((x_valid, input_data['x_valid']), axis=0)
# TC_valid = np.concatenate((TC_valid, input_data['TC_valid']), axis=0)
# y_valid = np.concatenate((y_valid, input_data['y_valid']), axis=0)
# y_cc_valid = np.concatenate((y_cc_valid, input_data['y_cc_valid']), axis=0)
# y_len_valid = np.concatenate((y_len_valid, input_data['y_len_valid']), axis=0)

print(np.shape(billboard_train_dataset['split_set']))
print(np.shape(aam_train_dataset['split_set']))

split_sets_train = np.concatenate((billboard_train_dataset['split_set'], aam_train_dataset['split_set']), axis=0)

print(np.shape(split_sets_train))

print("--------MERGE -----------")
train_dataset, valid_dataset = merge_data([billboard_file_dir, aam_file_dir])

print(np.shape(train_dataset['x']))
print(np.shape(train_dataset['split_set']))




