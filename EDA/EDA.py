import os
from collections import defaultdict
import csv
import re

def analyze_lab_files(directory, chord_count_file="chord_counts.csv", progression_file="chord_progressions.csv"):
    # Chord vocabulary
    chord_vocabulary = ['C:maj', 'C#:maj', 'D:maj', 'D#:maj', 'E:maj', 'F:maj', 'F#:maj', 'G:maj', 'G#:maj', 'A:maj', 'A#:maj', 'B:maj',
                        'C:min', 'C#:min', 'D:min', 'D#:min', 'E:min', 'F:min', 'F#:min', 'G:min', 'G#:min', 'A:min', 'A#:min', 'B:min',
                        'N', 'X']
    
    # Dictionary to store chord occurrences, chord progressions, and song lengths
    chord_count = defaultdict(int)
    chord_progression_count = defaultdict(int)
    song_lengths = []

    # Traverse the directory and its subdirectories to find .lab files
    for root, dirs, files in os.walk(directory):
        for file in files:

            #if file =! "majmin.lab":
            #    continue
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r') as f:
                previous_chord = None
                song_data = f.readlines()

                # If the file is empty, skip it
                if not song_data:
                    continue

                # Find the last non-empty line and extract the song length from it
                for line in reversed(song_data):
                    if line.strip():  # If the line is not empty
                        last_line = line.split()
                        song_length = float(last_line[1])  # End time from the last non-empty row
                        song_lengths.append(song_length)
                        break  # Exit the loop after finding the first non-empty line


                # Process each line to count chords and progressions
                previous_chord = None
                for line in song_data:
                    if line.strip(): # If the line is not empty
                        start, end, chord = line.strip().split()

                        # Filter out chords not in the chord vocabulary
                        if chord not in chord_vocabulary or chord == previous_chord:
                            continue

                        # Update chord occurrence
                        chord_count[chord] += 1

                        # Update chord progression (chord Y after chord Z, if they are different)
                        if previous_chord and previous_chord != chord:
                            progression = (previous_chord, chord)
                            chord_progression_count[progression] += 1
                        
                        # Update previous_chord for the next iteration
                        previous_chord = chord

    # Save chord counts to a CSV file
    with open(chord_count_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Chord', 'Count'])
        for chord, count in chord_count.items():
            writer.writerow([chord, count])
    
    # Save chord progressions to a CSV file (as a matrix)
    with open(progression_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header (all possible second chords)
        header = [''] + chord_vocabulary
        writer.writerow(header)

        # Write each row: first chord + the count for each second chord
        for chord1 in chord_vocabulary:
            row = [chord1]
            for chord2 in chord_vocabulary:
                row.append(chord_progression_count.get((chord1, chord2), 0))
            writer.writerow(row)

    # Calculate mean song length
    mean_song_length = sum(song_lengths) / len(song_lengths) if song_lengths else 0
    song_length_std = (sum((x - mean_song_length) ** 2 for x in song_lengths) / len(song_lengths)) ** 0.5 if song_lengths else 0


    # Return the results
    return {
        "chord_count_file": chord_count_file,
        "progression_file": progression_file,
        "mean_song_length": mean_song_length,
        "song_length_std" :  song_length_std
    }


# winterreise_dict = analyze_lab_files(directory=os.path.join('..', 'data', 'Winterreise', 'lab'), chord_count_file="winterreise_chord_counts.csv", progression_file="winterreise_chord_progressions.csv")
# print('Winterreise mean song length', str(winterreise_dict['mean_song_length']), 'std: ', str(winterreise_dict['song_length_std']))
aam_dict = analyze_lab_files(directory=os.path.join('..', '..', 'data', 'AAM', 'lab'), chord_count_file="aam_chord_counts.csv", progression_file="aam_chord_progressions.csv")
print('AAM mean song length', str(aam_dict['mean_song_length']), 'std: ', str(aam_dict['song_length_std']))
# billboard_dict = analyze_lab_files(directory=os.path.join('..', 'data', 'McGill-Billboard', 'McGill-Billboard-MIREX'), chord_count_file="billboard_chord_counts.csv", progression_file="billboard_chord_progressions.csv")
# print('Billboard mean song length', str(billboard_dict['mean_song_length']), 'std: ',  str(billboard_dict['song_length_std']))