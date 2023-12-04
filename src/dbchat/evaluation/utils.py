import csv
import json

import pandas as pd


def save_test_results(results, test_results_path):
    """Appends a json array with the results."""
    if (test_results_path).exists():
        with open(test_results_path, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.append(results)

    with open(test_results_path, 'a') as f:
        json.dump(results, f)


# Loader for evaluation run
# load config
# load evaluation data
# - input user queries
# - expected response
# - expected tables used
def load_evaluation_data(test_data_path, input_id):

    test_data = pd.read_csv(test_data_path, sep='|')
    test_data_expected = pd.read_csv(str(test_data_path).replace(
        'inputs', 'outputs'),
                                     sep='|')

    if input_id:
        input_query = test_data.loc[test_data.id == input_id, 'query']
        expected_tables = test_data.loc[test_data.id == input_id,
                                        'tables'].split(',')
        expected_response = test_data_expected.loc[
            test_data_expected.index == input_id,
            'response']  # "Berlin made $75.24"
        return input_query, expected_response, expected_tables

    return test_data.loc[:,
                         'query'], test_data_expected.loc[:,
                                                          'response'], test_data.loc[:,
                                                                                     'tables']


def evaluation_data_generator(test_data_path, input_id):
    test_data_file = str(test_data_path)
    test_data_expected_file = test_data_file.replace('inputs', 'outputs')

    fieldnames = {
        "input_query": "user_query",
        "expected_response": "response",
        "relevant_tables": "tables",
        "expected_query_id": "query_id",
    }

    with open(test_data_file, mode='r',
              newline='') as infile, open(test_data_expected_file,
                                          mode='r',
                                          newline='') as expectfile:

        reader = csv.DictReader(infile, delimiter='|')
        expected_reader = csv.DictReader(expectfile, delimiter='|')

        input_query, expected_response, relevant_tables = None, None, None
        if input_id:
            # Yield data for the specific ID
            for row in reader:
                if int(row['id']) == input_id:
                    input_query = row[fieldnames['input_query']]
                    relevant_tables = row[fieldnames['relevant_tables']].split(
                        ',')
                    break
            if input_query is None:
                raise ValueError(f"Input ID {input_id} not found in test data")

            for expected_row in expected_reader:
                if int(expected_row[
                        fieldnames['expected_query_id']]) == input_id:
                    expected_response = expected_row[
                        fieldnames['expected_response']]
                    break

            yield input_query, expected_response, relevant_tables
        else:
            # Yield all data
            for row, expected_row in zip(reader, expected_reader):
                input_query = row[fieldnames['input_query']]
                relevant_tables = row[fieldnames['relevant_tables']].split(',')
                expected_response = expected_row[
                    fieldnames['expected_response']]
                yield input_query, expected_response, relevant_tables


if __name__ == "__main__":
    print(
        next(
            evaluation_data_generator(
                "/mnt/c/Users/ssch7/repos/db-chat-assistant/src/tests/data/inputs/end-to-end.csv",
                input_id=1)))
