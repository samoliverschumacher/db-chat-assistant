import csv
import click


@click.command()
@click.option('--table_desc_file',
              help='Path to the existing table descriptions file.')
@click.option('--new_suffix', help='New suffix for the DOCUMENT_ID.')
@click.option(
    '--upload_file',
    help=
    'Path to the file with new entries to append. If not provided, entries will be read from table_desc_file.'
)
def main(table_desc_file, new_suffix, upload_file):
    """CLI script that adds new table descriptions to a table descriptions csv file."""

    new_entries = []
    seen_tables = set()

    if upload_file:
        # Open the file with new entries
        with open(upload_file, "r") as entries_file:
            reader = csv.reader(entries_file)

            # Skip the header row
            next(reader)

            # Iterate over the rows in the entries file
            for row in reader:
                table_name = row[0]
                description = row[1]

                if table_name not in seen_tables:
                    # Create the new row
                    document_id = f"{table_name}-{new_suffix}"
                    new_entry = [table_name, document_id, description]
                    new_entries.append(new_entry)
                    seen_tables.add(table_name)
    else:
        # Open the input CSV file
        with open(table_desc_file, "r") as input_file:
            reader = csv.reader(input_file)

            # Skip the header row
            next(reader)

            # Iterate over the rows in the input file
            for row in reader:
                table_name = row[0]

                if table_name not in seen_tables:
                    print(f"Adding description for table: {table_name}")

                    # Prompt the user for the new description
                    new_description = input("Enter the new description: ")

                    # Create the new row
                    document_id = f"{table_name}-{new_suffix}"
                    new_entry = [table_name, document_id, new_description]
                    new_entries.append(new_entry)
                    seen_tables.add(table_name)

    # Print out the accumulated new entries
    print("New entries:")
    for entry in new_entries:
        print(entry)

    # Prompt the user to confirm before appending to the file
    confirm = input(
        "Do you want to append the new entries to the file? (yes/no): ")
    if confirm.lower() == "yes":
        # Open the output CSV file in append mode
        with open(table_desc_file, "a", newline="") as output_file:
            writer = csv.writer(output_file)

            # Write the new entries to the output file
            for entry in new_entries:
                writer.writerow(entry)

        print("New entries appended to the file successfully!")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()
