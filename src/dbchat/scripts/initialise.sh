#!/bin/bash

#
# Description: This script creates metadata files for .csv files in a specified directory 
#              or the default "data" directory. Each metadata file contains the filename 
#              and the first row of the corresponding .csv file as the value for the "name" 
#              and "fields" attributes, respectively. The metadata files are stored in a subfolder 
#              named "metadata" within the specified directory or the default "data" directory.
# Usage: ./initialise.sh [directory]
# Arguments:
#   - directory (optional): The directory containing the .csv files. If not provided, a default 
#                           "data" directory two levels above the current directory will be used.
#


DEFAULT_DIR="$(dirname "$(dirname "$(realpath "$0")")")/data"  # expecting the src layout of a python project
DIR=${1:-"$DEFAULT_DIR"}  # Set default value to two directories higher than current directory in "data" folder

echo "Creating metadata files in $DIR"
# exit 0

# Create the metadata subfolder if it doesn't exist
mkdir -p "$DIR/metadata"

# Loop through all the .csv files in the specified directory
for file in "$DIR"/*.csv; do
    # Extract the filename without the extension
    filename="${file%.csv}"
    filename="${filename##*/}"
    
    # metadata file name
    metadata_file="$DIR/metadata/$filename.yaml"

    # Create the metadata file and write the content
    echo "name: \"$filename\"" | sed "s/\"//g" > "$metadata_file"
    echo -n "fields: " >> "$metadata_file"
    head -n 1 "$file" | sed "s/\"//g" >> "$metadata_file"
done