import numpy as np
import os
import itertools
import argparse
def convert_csv_to_lab(csv_dir, lab_dir, voca_type):

    valid_voca_types = ["shorthand", "extended", "majmin", "majmin_inv"]
    if voca_type not in valid_voca_types:
        raise ValueError("Voca type must be one of %r." % valid_voca_types)
    
    if not os.path.exists(lab_dir):
            os.makedirs(lab_dir)

    # process only for two performers that have audio available
    csv_files = [filename for filename in os.listdir(csv_dir) if ("HU33" in filename or  "SC06" in filename) ]
    for file_path in csv_files:
        starts = []
        ends = []
        chords = []
        with open(os.path.join(csv_dir,file_path)) as f:
            for line in itertools.islice(f, 1, None):  
                start, end, shorthand, extended, majmin, majmin_inv = line.split(";")
                starts.append(start)
                ends.append(end)
                if voca_type == "shorthand":
                    chords.append(shorthand[1:-1])
                elif voca_type == "extended":
                    chords.append(extended[1:-1])
                elif voca_type == "majmin":
                    chords.append(majmin[1:-1])
                elif voca_type == "majmin_inv":
                    chords.append(majmin_inv[1:-1])

        lab_file_name = file_path.replace(".csv", ".lab")        
        f = open(os.path.join(lab_dir, lab_file_name), "w")

        no_chords = len(starts)
        for i in range(no_chords):
            f.write(str(starts[i])+" "+str(ends[i])+" "+ chords[i])
            if i != no_chords - 1:
                f.write("\n")

        f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root directory for data', default='data/music/chord_recognition')
    parser.add_argument('--dataset_path', type=str, help='Name of dataset directory', default='Winterreise')
    parser.add_argument('--csv_dir_path', type=str, help='Directory in dataset directory with .arff files to be converted',default='ann_audio_chord')
    parser.add_argument('--lab_dir_path', type=str, help='Directory in dataset directory for the output .lab files to be stored', default='lab_majmin')
    parser.add_argument('--voca_type', type=str, help='Type of voca: shorthand, minmaj, extended or majmin_inv ', default='majmin')
    args = parser.parse_args()

    dataset_directory = os.path.join(args.root_path, args.dataset_path)
    csv_dir_path = os.path.join(dataset_directory, args.csv_dir_path)
    lab_dir_path = os.path.join(dataset_directory, args.lab_dir_path)

    convert_csv_to_lab(csv_dir_path, lab_dir_path, args.voca_type) 
    