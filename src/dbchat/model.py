import json
from pathlib import Path
import requests


def ask_orca_mini(prompt, config={}):
    """A very small low performance llm that can be run locally.
    
    To start the endpoint, after installation of ollama: `ollama serve`.
    """
    
    url = 'http://localhost:11434/api/generate'

    data = {
        "prompt": prompt,
        "model": "orca-mini",
        "stream": False
    }
    # Merge the additional data into the payload data
    data.update(config)
    
    response = requests.post(url, json=data)
    # Convert the response.text into a Python dictionary
    response_dict = json.loads(response.text)
    return response_dict["response"]


def ask_llama2_7B(prompt, config={}):
    """A very small low performance llm that can be run locally.
    
    To start the endpoint, after installation of ollama: `ollama serve`.
    """
    
    url = 'http://localhost:11434/api/generate'

    data = {
        "prompt": prompt,
        "model": "orca-mini",
        "stream": False
    }
    # Merge the additional data into the payload data
    data.update(config)
    
    response = requests.post(url, json=data)
    # Convert the response.text into a Python dictionary
    response_dict = json.loads(response.text)
    return response_dict["response"]


if __name__ == '__main__':
    # Load additional options from config.json
    config = {}
    if Path('config.json').exists():
        with open('config.json') as f:
            config = json.load(f)

    response = ask_orca_mini("Why is the sky blue?", 
                             config)
    print(response)
