import numpy as np
import librosa
import mir_eval
import torch
import os
from mir_eval.chord import validate, encode_many, rotate_bitmaps_to_roots

idx2chord = ['C:maj', 'C:min', 'C#:maj', 'C#:min', 'D:maj', 'D:min', 'D#:maj', 'D#:min', 'E:maj', 'E:min', 'F:maj', 'F:min', 'F#:maj',
             'F#:min', 'G:maj', 'G:min', 'G#:maj', 'G#:min', 'A:maj', 'A:min', 'A#:maj', 'A#:min', 'B:maj', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord

def audio_file_to_features(audio_file, config):
    original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    currunt_sec_hz = 0
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx
    tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=config.feature['n_bins'], bins_per_octave=config.feature['bins_per_octave'], hop_length=config.feature['hop_length'])
    feature = np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(original_wav)/config.mp3['song_hz']
    return feature, feature_per_second, song_length_second

# Audio files with format of wav and mp3
def get_audio_paths(audio_dir):
    return [os.path.join(root, fname) for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3') or fname.lower().endswith('.flac'))]


def chord_content_metric(reference_labels, estimated_labels):
    """Computes the Pitch Chroma Content metric from https://arxiv.org/abs/2201.05244

    Examples
    --------
    >>> (ref_intervals,
    ...  ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
    >>> (est_intervals,
    ...  est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
    >>> est_intervals, est_labels = mir_eval.util.adjust_intervals(
    ...     est_intervals, est_labels, ref_intervals.min(),
    ...     ref_intervals.max(), mir_eval.chord.NO_CHORD,
    ...     mir_eval.chord.NO_CHORD)
    >>> (intervals,
    ...  ref_labels,
    ...  est_labels) = mir_eval.util.merge_labeled_intervals(
    ...      ref_intervals, ref_labels, est_intervals, est_labels)
    >>> durations = mir_eval.util.intervals_to_durations(intervals)
    >>> comparisons = mir_eval.chord.chord_content_metric(ref_labels, est_labels)
    >>> score = mir_eval.chord.weighted_accuracy(comparisons, durations)

    Parameters
    ----------
    reference_labels : list, len=n
        Reference chord labels to score against.
    estimated_labels : list, len=n
        Estimated chord labels to score against.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        Comparison scores, in [0.0, 1.0, 0.6666667]

    """
    validate(reference_labels, estimated_labels)

    ref_data = encode_many(reference_labels, False)
    ref_chroma = rotate_bitmaps_to_roots(ref_data[1], ref_data[0])
    est_data = encode_many(estimated_labels, False)
    est_chroma = rotate_bitmaps_to_roots(est_data[1], est_data[0])

    # C is the number of predicted notes in the estimate that occur in the reference (ground truth)
    C = (ref_chroma * est_chroma).sum(axis=-1)

    # I is the number of insertions (extra predicted notes) in the estimate that are not present in 
    # the reference (ground truth)
    I = ((ref_chroma - est_chroma) == -1).sum(axis=-1)

    ref_len = ref_chroma.sum(axis=-1)

    # to avoid dividing by 0 for 'N' chord (scores are corrected later)
    ref_len = np.array([len if len!=0 else 1 for len in ref_len])
    
    # accuracy measurement for each chord estimate, scaled between 0 and 1
    scores = (C - I + ref_len) / (2* ref_len)

    # No-chord ('N') matching; match -1 roots,
    no_root = np.logical_and(ref_data[0] == -1, est_data[0] == -1)
    scores[no_root] = 1.0

    # 0 if only one is a 'N' chord
    zero_idx =  np.logical_xor(ref_data[0] == -1, est_data[0] == -1)
    scores[zero_idx] = 0.0

    # Ignore 'X' chords.
    skip_idx =  np.any(ref_data[1] < 0, axis=1)
    scores[skip_idx] = -1.0

    return scores

class metrics():
    def __init__(self):
        super(metrics, self).__init__()
        self.score_metrics = ['root', 'majmin', 'ccm']
        self.score_list_dict = dict()
        for i in self.score_metrics:
            self.score_list_dict[i] = list()
        self.average_score = dict()

    def score(self, gt_path, est_path):

        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)

        root_comparisons = mir_eval.chord.root(ref_labels, est_labels)
        root_score = mir_eval.chord.weighted_accuracy(root_comparisons, durations)

        majmin_comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
        majmin_score = mir_eval.chord.weighted_accuracy(majmin_comparisons, durations)

        ccm_comparisons = chord_content_metric(ref_labels, est_labels)
        ccm_score = mir_eval.chord.weighted_accuracy(ccm_comparisons, durations)

        self.score_list_dict["root"].append(root_score)
        self.score_list_dict["majmin"].append(majmin_score)
        self.score_list_dict["ccm"].append(ccm_score)


def lab_file_error_modify(ref_labels):
    for i in range(len(ref_labels)):
        if ref_labels[i][-2:] == ':4':
            ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
        elif ref_labels[i][-2:] == ':6':
            ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
        elif ref_labels[i][-4:] == ':6/2':
            ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
        elif ref_labels[i] == 'Emin/4':
            ref_labels[i] = 'E:min/4'
        elif ref_labels[i] == 'A7/3':
            ref_labels[i] = 'A:7/3'
        elif ref_labels[i] == 'Bb7/3':
            ref_labels[i] = 'Bb:7/3'
        elif ref_labels[i] == 'Bb7/5':
            ref_labels[i] = 'Bb:7/5'
        elif ref_labels[i].find(':') == -1:
            if ref_labels[i].find('min') != -1:
                ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
    return ref_labels

def root_majmin_ccm_score_calculation(valid_dataset, config, mean, std, device, model, verbose=False):
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()
    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            n_timestep = config.model['timestep']
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    encoder_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    prediction, _ = model.output_layer(encoder_output)
                    prediction = prediction.squeeze()
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time, time_unit * (n_timestep * t + i), idx2chord[prev_chord]))
                            break
            pid = os.getpid()
            tmp_path = os.path.join('tmp','tmp_' + str(pid) + '.lab')
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            score_metrics = ['root', 'majmin', 'ccm']
            metrics_.score(gt_path=lab_file_path, est_path=tmp_path)
            song_length_list.append(song_length_second)

            if verbose:
                for m in score_metrics:
                    print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))
        except:
            print('song name %s\' lab file error' % song_name)

    tmp = song_length_list / np.sum(song_length_list)
    for m in score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score