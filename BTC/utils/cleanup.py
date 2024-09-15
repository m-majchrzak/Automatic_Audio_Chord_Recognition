import argparse
import os

def delete_midis_rename_lab(folder_path):
    for filename in os.listdir(folder_path):
        if 'midi' in filename:
            os.remove(os.path.join(folder_path, filename))
        if 'lab' in filename:
            new_filename = filename.replace(".flac", "")
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, help='Root directory for data')
    args = parser.parse_args()
    delete_midis_rename_lab(args.folder_path)