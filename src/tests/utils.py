import json

SYNTHETIC_JUDGE_SYSTEM_PROMPT = """
Human: 
You are to rate a summarization on the scale of 0-10, with 0 being completely incorrect and 10 being a perfect summzarization of the given text.
Explain why you give your score.
Give a bullet point list of major differences between the reference and the summary.
I will supply a reference text by denoting REF: and the summarization to compare against with SUMMARY:.

REF:
{expected_response}

SUMMARY:
{response}

Assistant:"""


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
