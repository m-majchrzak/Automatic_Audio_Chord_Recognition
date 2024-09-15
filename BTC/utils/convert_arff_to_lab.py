import numpy as np
import os
from mutagen.mp3 import MP3
import itertools
import argparse
from flacduration import get_flac_duration
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

def chord_syntax_correct(chord, file_path):

    min_chords = [pitch+'min' for pitch in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']]
    maj_chords = [pitch+'maj' for pitch in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']]
    min_maj_chords = min_chords + maj_chords

    if chord == 'N.C.':
        return 'N'
    elif chord not in min_maj_chords:
        print('Unknown chord label converted do N: ', file_path, chord)
        return 'N'
    else:
        n = len(chord)
        pitch = chord[:n-3]
        min_maj = chord[-3:]
        chord_label = pitch + ':' + min_maj
    return(chord_label)

def convert_arff_to_lab(file_path, arff_dir_path, audio_dir_path, lab_dir_path, audio_type):

    valid_audio_types = {"mp3", "wav", "flac"}
    if audio_type not in valid_audio_types:
        raise ValueError("Convert arrf to lab: audio type must be one of %r." % valid_audio_types)

    starts = []
    ends = []
    chords = []
    first_line_flag = True

    with open(os.path.join(arff_dir_path,file_path)) as f:
        for line in itertools.islice(f, 7, None):  
            start, bar_count, quarter_count, chord = line.split(",")
            chord = chord[1:-2]

            starts.append(start)
            chord = chord_syntax_correct(chord, file_path)
            chords.append(chord)
            

            if first_line_flag:
                first_line_flag = False
            else:
                ends.append(start)

    
    song_name = file_path.replace("_beatinfo.arff", "_mix")
    if audio_type == "mp3":
        audio_path = os.path.join(audio_dir_path, song_name + ".mp3")
        audio = MP3(audio_path)
        last_end = audio.info.length
    elif audio_type == "wav":
        audio_path = os.path.join(audio_dir_path, song_name + ".wav")
        audio = WAVE(audio_path)
        last_end = audio.info.length
    elif audio_type == "flac":
        audio_path = os.path.join(audio_dir_path, song_name + ".flac")
        last_end = get_flac_duration(audio_path)

    # in case the last_end happens before the last start
    last_start = starts.pop()
    if float(last_end) >= float(last_start):
        starts.append(last_start)
        ends.append(last_end)
    else:
        print('Found reference label after end of song in : ', file_path)
        print('last start: ', last_start)
        print('song length: ', last_end)
    
    
    if not os.path.exists(lab_dir_path):
        os.makedirs(lab_dir_path)
    f = open(os.path.join(lab_dir_path, song_name+'.lab'), "w")

    no_chords = len(starts)
    for i in range(no_chords):
        f.write(str(starts[i])+" "+str(ends[i])+" "+ chords[i])
        if i != no_chords - 1:
            f.write("\n")

    f.close()


def convert_arff_to_lab_dir(arff_dir_path, audio_dir_path, lab_dir_path, audio_type):
    beatinfo_files = [filename for filename in os.listdir(arff_dir_path) if 'beatinfo' in filename]
    for file_path in beatinfo_files:
        convert_arff_to_lab(file_path, arff_dir_path, audio_dir_path, lab_dir_path, audio_type)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root directory for data', default='../data/music/chord_recognition')
    parser.add_argument('--dataset_path', type=str, help='Name of dataset directory', default='AAM')
    parser.add_argument('--arff_dir_path', type=str, help='Directory in dataset directory with .arff files to be converted',default='annotations')
    parser.add_argument('--audio_dir_path', type=str, help='Directory in dataset directory with audio files', default='audio-mixes-mp3')
    parser.add_argument('--lab_dir_path', type=str, help='Directory in dataset directory for the output .lab files to be stored', default='lab')
    parser.add_argument('--audio_type', type=str, help='Type of audio files: mp3, wav or flac', default='flac')
    args = parser.parse_args()

    dataset_directory = os.path.join(args.root_path, args.dataset_path)
    arff_dir_path = os.path.join(dataset_directory, args.arff_dir_path)
    audio_dir_path = os.path.join(dataset_directory, args.audio_dir_path)
    lab_dir_path = os.path.join(dataset_directory, args.lab_dir_path)

    convert_arff_to_lab_dir(arff_dir_path, audio_dir_path, lab_dir_path, args.audio_type) 
    