#!/bin/bash

# Search for .log files in the current directory and its subdirectories
log_files=($(find . -type f -name "*.log"))

# Prompt the user to select a log file
echo "Please select a log file:"
select log_file in "${log_files[@]}"; do
    if [[ -n $log_file ]]; then
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# Prompt the user for the log entry
echo "What entry would you like to add?"
read -r log_entry

# Check if log entry is empty
if [[ -z $log_entry ]]; then
    echo "Aborted"
    exit 0
fi

# Add the log entry to the selected file
echo "$(date) $(git config user.name):: $log_entry" >> "$log_file"

# Display the added log entry
echo "Log entry added: \"$(date) $(git config user.name): $log_entry\""