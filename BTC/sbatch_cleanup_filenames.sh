#!/bin/bash

# Check if directory is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Get the target directory
TARGET_DIR="$1"

# Check if the provided argument is a directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: $TARGET_DIR is not a directory"
  exit 1
fi

# Loop through all subdirectories in the target directory
for dir in "$TARGET_DIR"/*/; do
  # Check if it is a directory
  if [ -d "$dir" ]; then
    # Get the base name of the directory
    base_name=$(basename "$dir")
    # Construct the new directory name
    new_name="${base_name}_mix"
    # Rename the directory
    mv "$dir" "$TARGET_DIR/$new_name"
    echo "Renamed $dir to $TARGET_DIR/$new_name"
  fi
done