import os
import shutil

def create_folders_from_wav_files(input_directory, output_directory):
    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        print(f"The input directory {input_directory} does not exist.")
        return
    
    # Check if the output directory exists, create it if it does not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    # Iterate over each file in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file has a .wav extension
        if filename.endswith('.wav'):
            # Get the folder name by removing the .wav extension
            folder_name = filename[:-4]
            # Create the full path for the new folder in the output directory
            folder_path = os.path.join(output_directory, folder_name)
            
            # Create the folder if it does not exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")
            else:
                print(f"Folder already exists: {folder_path}")


def create_folders_and_copy_lab_files(input_directory, output_directory):
    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        print(f"The input directory {input_directory} does not exist.")
        return

    # Check if the output directory exists, create it if it does not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Iterate over each file in the input directory
    for filename in os.listdir(input_directory):
        # Skip directories
        if os.path.isdir(os.path.join(input_directory, filename)):
            continue
        
        # Get the folder name by removing the file extension
        folder_name = os.path.splitext(filename)[0]
        # Create the full path for the new folder in the output directory
        folder_path = os.path.join(output_directory, folder_name)

        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")

        # Copy the file to the new folder with the name 'majmin.lab'
        src_file_path = os.path.join(input_directory, filename)
        dst_file_path = os.path.join(folder_path, 'majmin.lab')
        shutil.copyfile(src_file_path, dst_file_path)
        print(f"Copied and renamed {filename} to {dst_file_path}")

def create_folders_and_copy_lab_files_aam(input_directory, output_directory):
    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        print(f"The input directory {input_directory} does not exist.")
        return

    # Check if the output directory exists, create it if it does not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Iterate over each file in the input directory
    for filename in os.listdir(input_directory):
        # Skip directories
        if os.path.isdir(os.path.join(input_directory, filename)):
            continue

        # Split the filename to extract the numeric prefix (xxxx)
        file_parts = filename.split('_')
        if len(file_parts) < 2 or not file_parts[0].isdigit():
            continue
        
        # Get the numeric part and convert it to an integer
        file_number = int(file_parts[0])
        
        # Check if the numeric part is between 0 and 192
        if 0 <= file_number <= 192:
            # Get the folder name by removing the file extension
            folder_name = os.path.splitext(filename)[0]
            # Create the full path for the new folder in the output directory
            folder_path = os.path.join(output_directory, folder_name)

            # Create the folder if it does not exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")

            # Copy the file to the new folder with the name 'majmin.lab'
            src_file_path = os.path.join(input_directory, filename)
            dst_file_path = os.path.join(folder_path, 'majmin.lab')
            shutil.copyfile(src_file_path, dst_file_path)
            print(f"Copied and renamed {filename} to {dst_file_path}")

if __name__ == "__main__":
    # Specify the input directory containing the .wav files
    #input_directory = "../../data/Winterreise/audio_wav"
    # Specify the output directory where the folders should be created
    #output_directory = "../../data/Winterreise/features"
    
    # Call the function to create folders
    #create_folders_from_wav_files(input_directory, output_directory)

    # Specify the input directory containing the files
    input_directory = "../../data/AAM/lab"
    # Specify the output directory where the folders should be created
    output_directory = "../data/AAM/lab"
    
    # Call the function to create folders and copy files
    create_folders_and_copy_lab_files_aam(input_directory, output_directory)