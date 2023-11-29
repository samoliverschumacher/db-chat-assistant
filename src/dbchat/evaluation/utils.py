import json


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
