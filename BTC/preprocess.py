import os
import librosa
from utils.chords import Chords
import re
from enum import Enum
import pyrubberband as pyrb
import torch
import math

class FeatureTypes(Enum):
    cqt = 'cqt'

class Preprocess():
    def __init__(self, config, feature_to_use, dataset_names, root_dir):
        self.config = config
        self.dataset_names = dataset_names
        self.root_path = root_dir + '/'

        self.time_interval = config.feature["hop_length"]/config.mp3["song_hz"]
        self.no_of_chord_datapoints_per_sequence = math.ceil(config.mp3['inst_len'] / self.time_interval)
        self.Chord_class = Chords()

	
        # artificial audio multitracks 
        self.aam_directory = self.root_path + 'AAM/'
        self.aam_audio_path = 'audio-mixes-mp3/'
        self.aam_lab_path = 'lab/'

        # winterreise
        self.winterreise_directory = self.root_path + 'Winterreise/'
        self.winterreise_audio_path = 'audio_wav/'
        self.winterreise_lab_path = 'lab_majmin/'

	

        self.feature_name = feature_to_use
        self.is_cut_last_chord = False

    def find_mp3_path(self, dirpath, word):
        for filename in os.listdir(dirpath):
            last_dir = dirpath.split("/")[-2]
            if ".mp3" in filename:
                tmp = filename.replace(".mp3", "")
                tmp = tmp.replace(last_dir, "")
                filename_lower = tmp.lower()
                filename_lower = " ".join(re.findall("[a-zA-Z]+", filename_lower))
                if word.lower().replace(" ", "") in filename_lower.replace(" ", ""):
                    return filename

    def get_all_files(self):
        res_list = []

          # artificial audio multitracks 
        if "aam" in self.dataset_names:
            # lab_files = os.listdir(os.path.join(self.aam_directory,self.aam_lab_path))
            lab_files = os.listdir(self.aam_directory + self.aam_lab_path)
            for filename in lab_files:
                song_name = filename.replace(".lab", "")
                mp3_path = song_name + ".flac"
                res_list.append([song_name, os.path.join(self.aam_directory, self.aam_lab_path, filename), os.path.join(self.aam_directory, self.aam_audio_path, mp3_path),
                                                                                    os.path.join("features", "aam")])   
        
        # winterreise 
        if "winterreise" in self.dataset_names:
            # lab_files = os.listdir(os.path.join(self.winterreise_directory,self.winterreise_lab_path))
            lab_files = os.listdir(self.winterreise_directory + self.winterreise_lab_path)
            for filename in lab_files:
                song_name = filename.replace(".lab", "")
                mp3_path = song_name + ".wav"
                res_list.append([song_name, os.path.join(self.winterreise_directory, self.winterreise_lab_path, filename), os.path.join(self.winterreise_directory, self.winterreise_audio_path, mp3_path),
                                                                                    os.path.join("features", "winterreise")])
        return res_list
    
    
    def song_pre(self, text):
        to_remove = ["'", '`', '(', ')', ' ', '&', 'and', 'And']

        for remove in to_remove:
            text = text.replace(remove, '')

        return text

    def config_to_folder(self):
        mp3_config = self.config.mp3
        feature_config = self.config.feature
        mp3_string = "%d_%.1f_%.1f" % \
                     (mp3_config['song_hz'], mp3_config['inst_len'],
                      mp3_config['skip_interval'])
        feature_string = "%s_%d_%d_%d" % \
                         (self.feature_name.value, feature_config['n_bins'], feature_config['bins_per_octave'], feature_config['hop_length'])

        return mp3_config, feature_config, mp3_string, feature_string

    def generate_labels_features_new(self, all_list):
        pid = os.getpid()
        mp3_config, feature_config, mp3_str, feature_str = self.config_to_folder()

        i = 0  # number of songs
        j = 0  # number of impossible songs
        k = 0  # number of tried songs
        total = 0  # number of generated instances

        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

        loop_broken = False
        for song_name, lab_path, mp3_path, save_path in all_list:

            # different song initialization
            if loop_broken:
                loop_broken = False

            i += 1
            print(pid, "generating features from ...", os.path.join(mp3_path))
            if i % 10 == 0:
                print(i, ' th song')

            original_wav, sr = librosa.load(os.path.join(mp3_path), sr=mp3_config['song_hz'])

            # make result path if not exists
            # save_path, mp3_string, feature_string, song_name, aug.pt

            result_path = os.path.join(save_path, mp3_str, feature_str, song_name.strip())
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            elif len(os.listdir(result_path)) != 0:
                print('Skipping preprocessing song:', song_name.strip())
                continue
	    
            # calculate result
            for stretch_factor in stretch_factors:
                if loop_broken:
                    loop_broken = False
                    break

                for shift_factor in shift_factors:
                    # for filename
                    idx = 0

                    chord_info = self.Chord_class.get_converted_chord(os.path.join(lab_path))

                    k += 1
                    # stretch original sound and chord info
                    x = pyrb.time_stretch(original_wav, sr, stretch_factor)
                    x = pyrb.pitch_shift(x, sr, shift_factor)
                    audio_length = x.shape[0]
                    chord_info['start'] = chord_info['start'] * 1/stretch_factor
                    chord_info['end'] = chord_info['end'] * 1/stretch_factor

                    # print(chord_info)
                    last_sec = chord_info.iloc[-1]['end']
                    last_sec_hz = int(last_sec * mp3_config['song_hz'])

                    if audio_length + mp3_config['skip_interval'] < last_sec_hz:
                        print('loaded song is too short :', song_name)
                        print('Audio_length:', audio_length)
                        print('Skip interval:', mp3_config['skip_interval'])
                        print('Last sec:', last_sec)
                        print('Last sec hz:', last_sec_hz)
                        #loop_broken = True
                        #j += 1
                        #break
                        

                    elif audio_length > last_sec_hz:
                        x = x[:last_sec_hz]

                    origin_length = last_sec_hz
                    origin_length_in_sec = origin_length / mp3_config['song_hz']

                    # print('origin_length_in_sec:', origin_length_in_sec)
                    # print('timeInterval: ', self.time_interval)

                    current_start_second = 0

                    # get chord list between current_start_second and current+song_length
                    while current_start_second + mp3_config['inst_len'] < origin_length_in_sec:
                        # print('current_start_second: ', current_start_second)
                        inst_start_sec = current_start_second
                        curSec = current_start_second

                        chord_list = []
                        # extract chord per 1/self.time_interval
                        while curSec < inst_start_sec + mp3_config['inst_len']:
                            try:
                                # print('curSec: ', curSec)
                                available_chords = chord_info.loc[(chord_info['start'] <= curSec) & (
                                        chord_info['end'] > curSec + self.time_interval)].copy()
                                

                                if len(available_chords) == 0:
                                    available_chords = chord_info.loc[((chord_info['start'] >= curSec) & (
                                            chord_info['start'] <= curSec + self.time_interval)) | (
                                                                              (chord_info['end'] >= curSec) & (
                                                                              chord_info['end'] <= curSec + self.time_interval))].copy()
                                if len(available_chords) == 1:
                                    chord = available_chords['chord_id'].iloc[0]
                                elif len(available_chords) > 1:
                                    max_starts = available_chords.apply(lambda row: max(row['start'], curSec),
                                                                        axis=1)
                                    available_chords['max_start'] = max_starts
                                    min_ends = available_chords.apply(
                                        lambda row: min(row.end, curSec + self.time_interval), axis=1)
                                    available_chords['min_end'] = min_ends
                                    chords_lengths = available_chords['min_end'] - available_chords['max_start']
                                    available_chords['chord_length'] = chords_lengths

                                    # Printing available_chords
                                    # print(available_chords)
                                    available_chords = available_chords.reset_index()
                                    chord = available_chords.iloc[available_chords['chord_length'].idxmax()]['chord_id'] # error
                                    
                                else:
                                    chord = 24

                                # print('chord:', chord)

                            except Exception as e:
                                chord = 24
                                print(e)
                                print(pid, "no chord")
                                raise RuntimeError()
                            
                            finally:
                                # convert chord by shift factor
                                if chord != 24:
                                    chord += shift_factor * 2
                                    chord = chord % 24

                                chord_list.append(chord)
                                curSec += self.time_interval

                        # print('first while finished')

                        if len(chord_list) == self.no_of_chord_datapoints_per_sequence:
                            try:
                                sequence_start_time = current_start_second
                                sequence_end_time = current_start_second + mp3_config['inst_len']

                                start_index = int(sequence_start_time * mp3_config['song_hz'])
                                end_index = int(sequence_end_time * mp3_config['song_hz'])

                                song_seq = x[start_index:end_index]

                                etc = '%.1f_%.1f' % (
                                    current_start_second, current_start_second + mp3_config['inst_len'])
                                aug = '%.2f_%i' % (stretch_factor, shift_factor)

                                if self.feature_name == FeatureTypes.cqt:
                                    # print(pid, "make feature")
                                    feature = librosa.cqt(song_seq, sr=sr, n_bins=feature_config['n_bins'],
                                                          bins_per_octave=feature_config['bins_per_octave'],
                                                          hop_length=feature_config['hop_length'])
                                else:
                                    raise NotImplementedError

                                if feature.shape[1] > self.no_of_chord_datapoints_per_sequence:
                                    feature = feature[:, :self.no_of_chord_datapoints_per_sequence]

                                if feature.shape[1] != self.no_of_chord_datapoints_per_sequence:
                                    print('loaded features length is too short :', song_name)
                                    loop_broken = True
                                    j += 1
                                    break

                                result = {
                                    'feature': feature,
                                    'chord': chord_list,
                                    'etc': etc
                                }

                                # save_path, mp3_string, feature_string, song_name, aug.pt
                                filename = aug + "_" + str(idx) + ".pt"
                                torch.save(result, os.path.join(result_path, filename))
                                idx += 1
                                total += 1
                            except Exception as e:
                                print(e)
                                print(pid, "feature error")
                                raise RuntimeError()
                        else:
                            print("invalid number of chord datapoints in sequence :", len(chord_list))
                        current_start_second += mp3_config['skip_interval']
        print(pid, "total instances: %d" % total)

    def generate_labels_features_voca(self, all_list):
        pid = os.getpid()
        mp3_config, feature_config, mp3_str, feature_str = self.config_to_folder()

        i = 0  # number of songs
        j = 0  # number of impossible songs
        k = 0  # number of tried songs
        total = 0  # number of generated instances
        stretch_factors = [1.0]
        shift_factors = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

        loop_broken = False
        for song_name, lab_path, mp3_path, save_path in all_list:
            save_path = save_path + '_voca'

            # different song initialization
            if loop_broken:
                loop_broken = False

            i += 1
            print(pid, "generating features from ...", os.path.join(mp3_path))
            if i % 10 == 0:
                print(i, ' th song')

            original_wav, sr = librosa.load(os.path.join(mp3_path), sr=mp3_config['song_hz'])

            # save_path, mp3_string, feature_string, song_name, aug.pt
            result_path = os.path.join(save_path, mp3_str, feature_str, song_name.strip())
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # calculate result
            for stretch_factor in stretch_factors:
                if loop_broken:
                    loop_broken = False
                    break

                for shift_factor in shift_factors:
                    # for filename
                    idx = 0

                    try:
                        chord_info = self.Chord_class.get_converted_chord_voca(os.path.join(lab_path))
                    except Exception as e:
                        print(e)
                        print(pid, " chord lab file error : %s" % song_name)
                        loop_broken = True
                        j += 1
                        break

                    k += 1
                    # stretch original sound and chord info
                    x = pyrb.time_stretch(original_wav, sr, stretch_factor)
                    x = pyrb.pitch_shift(x, sr, shift_factor)
                    audio_length = x.shape[0]
                    chord_info['start'] = chord_info['start'] * 1/stretch_factor
                    chord_info['end'] = chord_info['end'] * 1/stretch_factor

                    last_sec = chord_info.iloc[-1]['end']
                    last_sec_hz = int(last_sec * mp3_config['song_hz'])

                    if audio_length + mp3_config['skip_interval'] < last_sec_hz:
                        print('loaded song is too short :', song_name)
                        loop_broken = True
                        j += 1
                        break
                    elif audio_length > last_sec_hz:
                        x = x[:last_sec_hz]

                    origin_length = last_sec_hz
                    origin_length_in_sec = origin_length / mp3_config['song_hz']

                    current_start_second = 0

                    # get chord list between current_start_second and current+song_length
                    while current_start_second + mp3_config['inst_len'] < origin_length_in_sec:
                        inst_start_sec = current_start_second
                        curSec = current_start_second

                        chord_list = []
                        # extract chord per 1/self.time_interval
                        while curSec < inst_start_sec + mp3_config['inst_len']:
                            try:
                                available_chords = chord_info.loc[(chord_info['start'] <= curSec) & (chord_info['end'] > curSec + self.time_interval)].copy()
                                if len(available_chords) == 0:
                                    available_chords = chord_info.loc[((chord_info['start'] >= curSec) & (chord_info['start'] <= curSec + self.time_interval)) | ((chord_info['end'] >= curSec) & (chord_info['end'] <= curSec + self.time_interval))].copy()

                                if len(available_chords) == 1:
                                    chord = available_chords['chord_id'].iloc[0]
                                elif len(available_chords) > 1:
                                    max_starts = available_chords.apply(lambda row: max(row['start'], curSec),axis=1)
                                    available_chords['max_start'] = max_starts
                                    min_ends = available_chords.apply(lambda row: min(row.end, curSec + self.time_interval), axis=1)
                                    available_chords['min_end'] = min_ends
                                    chords_lengths = available_chords['min_end'] - available_chords['max_start']
                                    available_chords['chord_length'] = chords_lengths
                                    # print(available_chords)
                                    chord = available_chords['chord_id'][available_chords['chord_length'].idxmax()]
                                else:
                                    chord = 169
                            except Exception as e:
                                chord = 169
                                print(e)
                                print(pid, "no chord")
                                raise RuntimeError()
                            finally:
                                # convert chord by shift factor
                                if chord != 169 and chord != 168:
                                    chord += shift_factor * 14
                                    chord = chord % 168

                                chord_list.append(chord)
                                curSec += self.time_interval

                        if len(chord_list) == self.no_of_chord_datapoints_per_sequence:
                            try:
                                sequence_start_time = current_start_second
                                sequence_end_time = current_start_second + mp3_config['inst_len']

                                start_index = int(sequence_start_time * mp3_config['song_hz'])
                                end_index = int(sequence_end_time * mp3_config['song_hz'])

                                song_seq = x[start_index:end_index]

                                etc = '%.1f_%.1f' % (
                                    current_start_second, current_start_second + mp3_config['inst_len'])
                                aug = '%.2f_%i' % (stretch_factor, shift_factor)

                                if self.feature_name == FeatureTypes.cqt:
                                    feature = librosa.cqt(song_seq, sr=sr, n_bins=feature_config['n_bins'],
                                                          bins_per_octave=feature_config['bins_per_octave'],
                                                          hop_length=feature_config['hop_length'])
                                else:
                                    raise NotImplementedError

                                if feature.shape[1] > self.no_of_chord_datapoints_per_sequence:
                                    feature = feature[:, :self.no_of_chord_datapoints_per_sequence]

                                if feature.shape[1] != self.no_of_chord_datapoints_per_sequence:
                                    print('loaded features length is too short :', song_name)
                                    loop_broken = True
                                    j += 1
                                    break

                                result = {
                                    'feature': feature,
                                    'chord': chord_list,
                                    'etc': etc
                                }

                                # save_path, mp3_string, feature_string, song_name, aug.pt
                                filename = aug + "_" + str(idx) + ".pt"
                                torch.save(result, os.path.join(result_path, filename))
                                idx += 1
                                total += 1
                            except Exception as e:
                                print(e)
                                print(pid, "feature error")
                                raise RuntimeError()
                        else:
                            print("invalid number of chord datapoints in sequence :", len(chord_list))
                        current_start_second += mp3_config['skip_interval']
        print(pid, "total instances: %d" % total)