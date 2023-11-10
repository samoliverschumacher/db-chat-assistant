# db-chat-assistant
connects a database schema information to a LLM to give results without knowing SQL

# The Problem
- People without SQL or tableau knowledge can't make data driven decisions on their own.

---

[Schematic for this system](https://app.diagrams.net/#G1JNsvPrjDpMweYzJn9UKzID2g3g04YQyP)


# Installation

## Project

Create environment (is langauge agnostic, tools like pip to install python deps)
```bash
conda env create -f environment.yml
```

Instal python deps, including development and test related dependencies
```bash
pip install -e .[test,dev]
```

```bash
pre-commit install
```
## Ollama - Running LLMs locally

More info here: https://python.langchain.com/docs/guides/local_llms

[Install instructions from repo.](https://github.com/jmorganca/ollama)
```
curl https://ollama.ai/install.sh | sh
```
First;
```
ollama serve
```

In a separate terminal;
```
ollama run orca-mini
```
