from utils.dataset_utils import load_data
import os
import numpy as np
import pickle

# dataset_name = "winterreise"
# example_song_name = 'Schubert_D911-02_HU33'

dataset_name = "billboard"
example_song_name = '1269'

print("------ AFTER READ ----------")
inputdir = os.path.join('preprocessed_data', dataset_name, f'{dataset_name}_data.pkl')
with open(inputdir, 'rb') as input_data:
    data = pickle.load(input_data)

print(np.shape(data[example_song_name]))
print(data[example_song_name][100])

print("------ AFTER AUGMENT ----------")
inputdir = os.path.join('preprocessed_data', dataset_name, f'{dataset_name}_data_shift_' + str(1) + '.pkl')
with open(inputdir, 'rb') as input_data:
        data_shift = pickle.load(input_data)

print(np.shape(data_shift[example_song_name]['chroma']))
print(np.shape(data_shift[example_song_name]['TC']))
print(np.shape(data_shift[example_song_name]['chord']))
print(np.shape(data_shift[example_song_name]['chordChange']))
print(data_shift[example_song_name])

print("------ AFTER SEGMENT ----------")

inputdir = os.path.join('preprocessed_data', dataset_name, f'{dataset_name}_data_shift_segment_' + str(0) + '.pkl') # with segment
with open(inputdir, 'rb') as input_data:
    data_shift = pickle.load(input_data)

print(np.shape(data_shift[example_song_name]['chroma']))
print(np.shape(data_shift[example_song_name]['TC']))
print(np.shape(data_shift[example_song_name]['chord']))
print(np.shape(data_shift[example_song_name]['chordChange']))
print(data_shift[example_song_name])

n_steps = 100
data_reshape = {}
for key, value in data_shift.items():
    if key == example_song_name:
        print(f"---------{key}----------------")
        chroma = value['chroma']
        TC = value['TC']
        chord = value['chord']
        chordChange = value['chordChange']

        n_frames = chroma.shape[0]
        print(f"n_frames: {n_frames}")
        n_pad = 0 if n_frames/n_steps == 0 else n_steps - (n_frames % n_steps)
        print(f"n_pad: {n_pad}")
        if n_pad != 0: # chek if need paddings
            chroma = np.pad(chroma, [(0, n_pad), (0, 0)], 'constant', constant_values=0)
            TC = np.pad(TC, [(0, n_pad), (0, 0)], 'constant', constant_values=0)
            chord = np.pad(chord, [(0, n_pad)], 'constant', constant_values=24) # 24 for padding frams
            chordChange = np.pad(chordChange, [(0, n_pad)], 'constant', constant_values=0) # 0 for padding frames

        seq_hop = n_steps // 2
        print(f"seq_hop: {seq_hop}")
        print(f"chroma.shape[0]: {chroma.shape[0]}")
        n_sequences = int((chroma.shape[0] - n_steps) / seq_hop) + 1
        print(f"n_sequences: {n_sequences}")
        _, feature_size = chroma.shape
        print(f"feature_size: {feature_size}")
        _, TC_size = TC.shape
        print(f"TC_size: {TC_size}")
        s0, s1 = chroma.strides
        print(f"s0: {s0}, s1: {s1}")
        chroma_reshape = np.lib.stride_tricks.as_strided(chroma, shape=(n_sequences, n_steps, feature_size), strides=(s0 * seq_hop, s0, s1))
        ss0, ss1 = TC.strides
        TC_reshape = np.lib.stride_tricks.as_strided(TC, shape=(n_sequences, n_steps, TC_size), strides=(ss0 * seq_hop, ss0, ss1))
        sss0, = chord.strides
        chord_reshape = np.lib.stride_tricks.as_strided(chord, shape=(n_sequences, n_steps), strides=(sss0 * seq_hop, sss0))
        ssss0, = chordChange.strides
        chordChange_reshape = np.lib.stride_tricks.as_strided(chordChange, shape=(n_sequences, n_steps), strides=(ssss0 * seq_hop, ssss0))
        sequenceLen = np.array([n_steps for _ in range(n_sequences - 1)] + [n_steps - n_pad], dtype=np.int32) # [n_sequences]

        """data_reshape = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array, 'sequenceLen': array, 'nSequence': array}, ...}"""
        data_reshape[key] = {}
        data_reshape[key]['chroma'] = chroma_reshape
        data_reshape[key]['TC'] = TC_reshape
        data_reshape[key]['chord'] = chord_reshape
        data_reshape[key]['chordChange'] = chordChange_reshape
        data_reshape[key]['sequenceLen'] = sequenceLen
        data_reshape[key]['nSequence'] = n_sequences

print("------ AFTER RESHAPE ----------")

inputdir = os.path.join('preprocessed_data', dataset_name, f'{dataset_name}_data_reshape_0.pkl') # with segment
with open(inputdir, 'rb') as input_data:
    data_reshape_0 = pickle.load(input_data)

print(np.shape(data_reshape_0[example_song_name]['chroma']))
print(np.shape(data_reshape_0[example_song_name]['TC']))
print(np.shape(data_reshape_0[example_song_name]['chord']))
print(np.shape(data_reshape_0[example_song_name]['chordChange']))
print(np.shape(data_reshape_0[example_song_name]['sequenceLen']))
print(np.shape(data_reshape_0[example_song_name]['nSequence']))
print(data_reshape_0[example_song_name])

print(np.shape(data_reshape_0[example_song_name]['chroma'][::2]))

# split_sets = {}
# split_sets['valid'] = [('Schubert_D911-01_HU33', 7), ('Schubert_D911-01_SC06', 8), (example_song_name, 3), ('Schubert_D911-02_SC06', 3), ('Schubert_D911-03_HU33', 4), ('Schubert_D911-03_SC06', 3), ('Schubert_D911-04_HU33', 4), ('Schubert_D911-04_SC06', 4)]
# print(np.shape(np.concatenate([data_reshape_0[info[0]]['chroma'][::2] for info in split_sets['valid']], axis=0)))

print("------ AFTER SPLIT ----------")
train_dataset, valid_dataset = load_data(os.path.join('preprocessed_data', dataset_name, f'{dataset_name}_data_model_input_final_0_192_1.npz'))

print("---------TRAIN--------------")
print(f"x: {np.shape(train_dataset['x'])}")
print(f"TC: {np.shape(train_dataset['TC'])}")
print(f"y: {np.shape(train_dataset['y'])}")
print(f"y_cc: {np.shape(train_dataset['y_cc'])}")
print(f"y_len: {np.shape(train_dataset['y_len'])}")
print(train_dataset['split_set'])


print("---------VALID--------------")
print(f"x: {np.shape(valid_dataset['x'])}")
print(f"TC: {np.shape(valid_dataset['TC'])}")
print(f"y: {np.shape(valid_dataset['y'])}")
print(f"y_cc: {np.shape(valid_dataset['y_cc'])}")
print(f"y_len: {np.shape(valid_dataset['y_len'])}")
print(valid_dataset['split_set'])