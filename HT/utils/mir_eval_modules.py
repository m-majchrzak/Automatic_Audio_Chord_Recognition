import numpy as np
import librosa
import mir_eval
import torch
import os
from mir_eval.chord import validate, encode_many, rotate_bitmaps_to_roots
from .dataset_utils import get_files, get_song_length

idx2chord = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj', 'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min','N', 'X']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

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

    def score(self, ref_labels, est_labels, durations):

        root_comparisons = mir_eval.chord.root(ref_labels, est_labels)
        root_score = mir_eval.chord.weighted_accuracy(root_comparisons, durations)

        majmin_comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
        majmin_score = mir_eval.chord.weighted_accuracy(majmin_comparisons, durations)

        ccm_comparisons = chord_content_metric(ref_labels, est_labels)
        ccm_score = mir_eval.chord.weighted_accuracy(ccm_comparisons, durations)

        return [root_score, majmin_score, ccm_score]
        



def root_majmin_ccm_score_calculation(dataset_name, valid_dataset, sess, chord_predictions, x, y, y_cc, y_len, dropout_rate, is_training, slope, stochastic_tensor, verbose=False):

    # Inference using the model
    val_feed_dict = {
        x: valid_dataset['x'],
        y_cc: valid_dataset['y_cc'],
        y: valid_dataset['y'],
        y_len: valid_dataset['y_len'],
        dropout_rate: 0.0,
        is_training: False,
        slope: 1.0,
        stochastic_tensor: False
    }
    predictions = sess.run(chord_predictions, feed_dict=val_feed_dict)

    segment_duration_dict = {'billboard': 0.046439909*21, 'aam' : 0.046439909*21, 'winterreise': 0.046439909*21}
    split_set = valid_dataset['split_set']
    score_metrics = ['root', 'majmin', 'ccm']
    metrics_ = metrics()
    song_length_list = list()

    for index, (song_name, n_seq) in enumerate(split_set):
        song_name, lab_file_path, feature_file_path = get_files(dataset_name, song_name)
        song_length_second = get_song_length(dataset_name, feature_file_path)
        print(f'song_length_second: {song_length_second}')
        
        song_root_scores_list = []
        song_majmin_scores_list = []
        song_ccm_scores_list = []

        for i_seq in range(n_seq):
            ref_labels = [idx2chord[label] for label in valid_dataset['y'][index+i_seq]]
            est_labels = [idx2chord[label] for label in predictions[index+i_seq]]
            durations = np.repeat([segment_duration_dict[dataset_name]], len(valid_dataset['y'][index+i_seq]))

            scores = metrics_.score(ref_labels=ref_labels, est_labels=est_labels, durations=durations)
            song_root_scores_list.append(scores[0])
            song_majmin_scores_list.append(scores[1])
            song_ccm_scores_list.append(scores[2])

        metrics_.score_list_dict["root"].append(np.sum(song_root_scores_list)/n_seq)
        metrics_.score_list_dict["majmin"].append(np.sum(song_majmin_scores_list)/n_seq)
        metrics_.score_list_dict["ccm"].append(np.sum(song_ccm_scores_list)/n_seq)
        song_length_list.append(song_length_second)

        if verbose:
            for m in score_metrics:
                print('song name %s, %s score : %.4f' % (song_name, m, metrics_.score_list_dict[m][-1]))

    tmp = song_length_list / np.sum(song_length_list)
    for m in score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score