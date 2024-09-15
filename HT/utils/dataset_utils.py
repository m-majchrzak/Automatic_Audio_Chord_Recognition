import numpy as np
import os
import csv

def load_data(file_dir):
    with np.load(file_dir, allow_pickle=True) as input_data:
        split_sets = input_data['split_sets'].item()

        train_dataset = {
            "x" : input_data['x_train'],
            "TC" : input_data['TC_train'],
            "y" : input_data['y_train'],
            "y_cc" : input_data['y_cc_train'],
            "y_len" : input_data['y_len_train'],
            "split_set": split_sets['train']
        }

        valid_dataset = {
            "x" : input_data['x_valid'],
            "TC" : input_data['TC_valid'],
            "y" : input_data['y_valid'],
            "y_cc" : input_data['y_cc_valid'],
            "y_len" : input_data['y_len_valid'],
            "split_set": split_sets['valid']
        }
        
    return train_dataset, valid_dataset


def merge_data(file_dir_list):
    file_dir = file_dir_list[0]
    train_dataset, valid_dataset = load_data(file_dir)
    for i in range(1,len(file_dir_list)):
        with np.load(file_dir_list[i], allow_pickle=True) as input_data:
            split_sets = input_data['split_sets'].item()

            print('Current train split set:', train_dataset['split_set'])
            print('New train split set:', split_sets['train'])
 
            train_dataset = {
                "x" : np.concatenate((train_dataset['x'], input_data['x_train']), axis=0),
                "TC" : np.concatenate((train_dataset['TC'], input_data['TC_train']), axis=0),
                "y" : np.concatenate((train_dataset['y'], input_data['y_train']), axis=0),
                "y_cc" : np.concatenate((train_dataset['y_cc'], input_data['y_cc_train']), axis=0),
                "y_len" : np.concatenate((train_dataset['y_len'], input_data['y_len_train']), axis=0),
                "split_set": np.concatenate((train_dataset['split_set'], split_sets['train']), axis=0)
            }
            valid_dataset = {
                "x" : np.concatenate((valid_dataset['x'], input_data['x_valid']), axis=0),
                "TC" : np.concatenate((valid_dataset['TC'], input_data['TC_valid']), axis=0),
                "y" : np.concatenate((valid_dataset['y'], input_data['y_valid']), axis=0),
                "y_cc" : np.concatenate((valid_dataset['y_cc'], input_data['y_cc_valid']), axis=0),
                "y_len" : np.concatenate((valid_dataset['y_len'], input_data['y_len_valid']), axis=0),
                "split_set": np.concatenate((valid_dataset['split_set'], split_sets['valid']), axis=0)
            }
        
    return train_dataset, valid_dataset


def get_all_files(valid_dataset, dataset_name):
    res_list = []
    dataset_dir_dict = {'billboard': 'McGill-Billboard', 'aam' : 'AAM', 'winterreise': 'Winterreise'}
    lab_dir_dict = {'billboard': "McGill-Billboard-MIREX", 'aam' : 'lab', 'winterreise': 'lab'}
    feature_dir_dict = {'billboard': "McGill-Billboard-Features", 'aam' : 'feature', 'winterreise': 'feature'}
    dataset_directory = dataset_dir_dict[dataset_name]
    lab_directory = lab_dir_dict[dataset_name]
    feature_directory = feature_dir_dict[dataset_name]

    song_names = os.listdir(os.path.join('..', 'data', dataset_directory, lab_directory))
    for song_name in song_names:
        res_list.append([song_name, 
                         os.path.join("data", dataset_directory, lab_directory, song_name, "majmin.lab"), #lab
                         os.path.join("data", dataset_directory, feature_directory, song_name, "bothchroma.csv")]) #feature
    return res_list

def get_files(dataset_name, song_name):
    dataset_dir_dict = {'billboard': 'McGill-Billboard', 'aam' : 'AAM', 'winterreise': 'Winterreise'}
    lab_dir_dict = {'billboard': "McGill-Billboard-MIREX", 'aam' : 'lab', 'winterreise': 'lab'}
    feature_dir_dict = {'billboard': "McGill-Billboard-Features", 'aam' : 'features', 'winterreise': 'features'}
    dataset_directory = dataset_dir_dict[dataset_name]
    lab_directory = lab_dir_dict[dataset_name]
    feature_directory = feature_dir_dict[dataset_name]

    return [song_name, 
            os.path.join("data", dataset_directory, lab_directory, song_name, "majmin.lab"), #lab
            os.path.join("data", dataset_directory, feature_directory, song_name, "bothchroma.csv")] #feature

def get_song_length(dataset_name, feature_file):
    with open(feature_file, mode='r') as file:
        reader = csv.reader(file, delimiter=',')
        last_row = None
        # Iterate over all rows to reach the last one
        for row in reader:
            last_row = row
        # If there are rows in the file, return the first value of the last row
        if last_row:
            if dataset_name == 'billboard':
                return float(last_row[1])
            else:
                return float(last_row[0])
        else:
            return None
