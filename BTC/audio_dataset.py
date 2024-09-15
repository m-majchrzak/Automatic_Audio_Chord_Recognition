import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import Preprocess, FeatureTypes
import math
from multiprocessing import Pool
from sortedcontainers import SortedList

class AudioDataset(Dataset):
    def __init__(self, config, root_dir='data/music/chord_recognition', dataset_name='aam', subset=3000, multiplier=1,
                 featuretype=FeatureTypes.cqt, num_workers=20, train=False, preprocessing=False, resize=None, kfold=0, total_fold=6):
        super(AudioDataset, self).__init__()

        self.config = config
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.subset = subset
        self.multiplier = multiplier
        self.preprocessor = Preprocess(config, featuretype, dataset_name, self.root_dir)
        self.resize = resize
        self.train = train
        self.ratio = config.experiment['data_ratio']

        # preprocessing hyperparameters
        # song_hz, n_bins, bins_per_octave, hop_length
        mp3_config = config.mp3
        feature_config = config.feature
        self.mp3_string = "%d_%.1f_%.1f" % \
                          (mp3_config['song_hz'], mp3_config['inst_len'],
                           mp3_config['skip_interval'])
        self.feature_string = "%s_%d_%d_%d" % \
                              (featuretype.value, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])

        
        is_preprocessed = True if os.path.exists(os.path.join('features', dataset_name, self.mp3_string, self.feature_string)) else False
        
        print('Is preprocessed:'+str(is_preprocessed))

        if (not is_preprocessed) | preprocessing:
            midi_paths = self.preprocessor.get_all_files()
            print("Midi files len:"+str(len(midi_paths)))

            if num_workers > 1:
                num_path_per_process = math.ceil(len(midi_paths) / num_workers)
                args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process]
                        for i in range(num_workers)]

                # start process
                p = Pool(processes=num_workers)
                p.map(self.preprocessor.generate_labels_features_new, args)

                p.close()
            else:
                self.preprocessor.generate_labels_features_new(midi_paths)

        self.song_names, self.paths = self.get_paths(kfold=kfold, total_fold=total_fold)

        print('Song names:')
        print(self.song_names)

        print('Paths:')
        print(self.paths)

        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        instance_path = self.paths[idx]

        res = dict()
        data = torch.load(instance_path)
        res['feature'] = np.log(np.abs(data['feature']) + 1e-6)
        res['chord'] = data['chord']
        return res

    def get_paths(self, kfold, total_fold):
        temp = {}
        used_song_names = list()
        dataset_path = os.path.join("features", self.dataset_name, self.mp3_string, self.feature_string)
        # print(dataset_path)
        song_names = np.sort(os.listdir(dataset_path))[0:self.subset]
        for song_name in song_names:
            paths = []
            instance_names = os.listdir(os.path.join(dataset_path, song_name))
            if len(instance_names) > 0:
                used_song_names.append(song_name)
            for instance_name in instance_names:
                paths.append(os.path.join(dataset_path, song_name, instance_name))
            temp[song_name] = paths
                    
        song_names = used_song_names
        song_names = SortedList(song_names)

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
        print('Total instances (train and valid) : %d' %len(tmp))

        # divide train/valid dataset using k fold
        result = []
        quotient = len(song_names) // total_fold
        remainder = len(song_names) % total_fold
        fold_num = [0]
        for i in range(total_fold):
            fold_num.append(quotient)
        for i in range(remainder):
            fold_num[i+1] += 1
        for i in range(total_fold):
                fold_num[i+1] += fold_num[i]

        if self.train:
            tmp = []
            # get not augmented data
            for k in range(total_fold):
                if k != kfold:
                    for i in range(fold_num[k], fold_num[k+1]):
                        result += temp[song_names[i]]
                    tmp += song_names[fold_num[k]:fold_num[k + 1]]
            song_names = tmp

            # train multiplication
            song_names = np.repeat(song_names, self.multiplier)
            result = np.repeat(result, self.multiplier)
        else:
            for i in range(fold_num[kfold], fold_num[kfold+1]):
                instances = temp[song_names[i]]
                instances = [inst for inst in instances if "1.00_0" in inst] 
                result += instances
            song_names = song_names[fold_num[kfold]:fold_num[kfold+1]]
            
        return song_names, result
    
    def get_paths_combined(self, aam_train_number = 200, aam_duplication=1, aam_test_number=2,
                        winterreise_train_number = 44, winterreise_duplication=4, winterreise_test_number=2):
        temp = {}
        used_song_names = list()

        aam_dataset_path = os.path.join("features", "aam", self.mp3_string, self.feature_string)
        winterreise_dataset_path = os.path.join("features", "winterreise", self.mp3_string, self.feature_string)

        if self.train:
            aam_song_names = os.listdir(aam_dataset_path)[0:aam_train_number]
            aam_song_names = np.repeat(aam_song_names, aam_duplication)

            winterreise_song_names = np.sort(os.listdir(winterreise_dataset_path))[0:winterreise_train_number]
            winterreise_song_names = np.repeat(winterreise_song_names, winterreise_duplication)
        else:
            aam_song_names = os.listdir(aam_dataset_path)[aam_train_number:aam_train_number+aam_test_number]
            print("Test aam names:", aam_song_names)
            winterreise_song_names = np.sort(os.listdir(winterreise_dataset_path))[winterreise_train_number:winterreise_train_number+winterreise_test_number]
            print("Test winterreise names:", winterreise_song_names)

        print("Length of aam songs:", len(aam_song_names))
        print("Length of winterreise songs:", len(winterreise_song_names))
        
        for song_name in aam_song_names:
            paths = []
            instance_names = os.listdir(os.path.join(aam_dataset_path, song_name))
            if len(instance_names) > 0:
                used_song_names.append(song_name)
            else:
                #print("No instance for song:", song_name, "path: ", os.listdir(os.path.join(aam_dataset_path, song_name)))
                raise Exception("No instance for song:", song_name, "path: ", os.listdir(os.path.join(aam_dataset_path, song_name)))
            for instance_name in instance_names:
                paths.append(os.path.join(aam_dataset_path, song_name, instance_name))
            temp[song_name] = paths
            #result += paths

        for song_name in winterreise_song_names:
            paths = []
            instance_names = os.listdir(os.path.join(winterreise_dataset_path, song_name))
            if len(instance_names) > 0:
                used_song_names.append(song_name)
            else:
                #print("No instance for song:", song_name, "path: ", os.listdir(os.path.join(winterreise_dataset_path, song_name)))
                raise Exception("No instance for song:", song_name, "path: ", os.listdir(os.path.join(winterreise_dataset_path, song_name)))
            for instance_name in instance_names:
                paths.append(os.path.join(winterreise_dataset_path, song_name, instance_name))
            temp[song_name] = paths
            # result += paths

        print('Total song length : %d' %len(used_song_names))

        song_names = used_song_names
        song_names = SortedList(song_names)

        result = []

        print('Total used song length : %d' %len(song_names))
        tmp = []
        for i in range(len(song_names)):
            tmp += temp[song_names[i]]
            instances = temp[song_names[i]]
            instances = [inst for inst in instances if "1.00_0" in inst] ## 1.00_0 in instance name??
            result += instances
        print('Total instances (train and valid) : %d' %len(tmp))

        return song_names, result

def _collate_fn(batch):
    batch_size = len(batch)
    max_len = batch[0]['feature'].shape[1]

    input_percentages = torch.empty(batch_size)  # for variable length
    chord_lens = torch.empty(batch_size, dtype=torch.int64)
    chords = []
    collapsed_chords = []
    features = []
    boundaries = []
    for i in range(batch_size):
        sample = batch[i]
        feature = sample['feature']
        chord = sample['chord']
        diff = np.diff(chord, axis=0).astype(bool)
        idx = np.insert(diff, 0, True, axis=0)
        chord_lens[i] = np.sum(idx).item(0)
        chords.extend(chord)
        features.append(feature)
        input_percentages[i] = feature.shape[1] / max_len
        collapsed_chords.extend(np.array(chord)[idx].tolist())
        boundary = np.append([0], diff)
        boundaries.extend(boundary.tolist())

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # batch_size*1*feature_size*max_len
    chords = torch.tensor(chords, dtype=torch.int64)  # (batch_size*time_length)
    collapsed_chords = torch.tensor(collapsed_chords, dtype=torch.int64)  # total_unique_chord_len
    boundaries = torch.tensor(boundaries, dtype=torch.uint8)  # (batch_size*time_length)

    return features, input_percentages, chords, collapsed_chords, chord_lens, boundaries

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
