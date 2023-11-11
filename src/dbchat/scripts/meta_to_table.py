import os
import csv
import yaml


# Get the label input from the user
label = input("Enter the label: ")
folder_path = input("Enter the metadata directory containing the table.yaml files: ") or "metadata"
output_file = input("Enter the output_file name: ") or "table_descriptions.csv"

# Create the output CSV file and write the header row
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["TABLE_NAME", "DOCUMENT_ID", "DESCRIPTION"])

    # Iterate over the YAML files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            file_path = os.path.join(folder_path, filename)

            # Load the YAML file
            with open(file_path) as yaml_file:
                data = yaml.safe_load(yaml_file)

            # Extract the relevant information from the YAML data
            table_name = data.get("name")
            description = data.get("description")

            # Create the document ID by combining the table name and label
            document_id = f"{table_name}-{label}"

            # Write the row to the CSV file
            writer.writerow([table_name, document_id, description])

print("Conversion of YAML metadata fiels to CSVcompleted successfully!")