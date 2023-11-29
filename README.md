# db-chat-assistant
connects a database schema information to a LLM to give results without knowing SQL

# The Problem
- People without SQL or tableau knowledge can't make data driven decisions on their own.

---

[Schematic for this system](https://app.diagrams.net/#G1JNsvPrjDpMweYzJn9UKzID2g3g04YQyP)


# Installation

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

In a separate terminal, download a model.

Smallest (1.9GB)
```
ollama run orca-mini
```

Compatible with llama-index (3.8GB)
```
ollama run llama2
```

### Customise a local model

- See [example in Modelfile](src/dbchat/models/ollama-llama2-reranker) for more info
- See parameters in https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md


## SQLLite CSV database locally.

```
sudo apt install sqlite3
```



### Create metadata from a database file

Create metadata YAML files from each tables' CSV file, into a `src/dbchat/data/metadata` folder.
```bash
src/dbchat/scripts/initialise.sh src/dbchat/data
```

Convert the YAML table metadata files into rows of a single `table_descriptions.csv`
```bash
python src/dbchat/scripts/meta_to_table.py
```

Add new metadata descriptions to the `table_descriptions.csv` using this CLI script
```bash
python src/dbchat/scripts/add_table_descriptions.py
```

Add a new table to the database that contains a row for each table's metadata
```
> sqlite3 data/database.db
> CREATE TABLE table_descriptions (
       TABLE_NAME TEXT,
       DOCUMENT_ID TEXT,
       DESCRIPTION TEXT
   );
> .mode csv
> .import '| tail -n +2 data/table_descriptions.csv' table_descriptions
> SELECT * FROM table_descriptions;
```

### Create a vector store for the db metadata descriptions

```python
from llama_index import download_loader
from sqlalchemy import create_engine
def load_metadata_from_sqllite():
    DatabaseReader = download_loader("DatabaseReader")

    engine = create_engine(f"sqlite:///{db_path}")
    reader = DatabaseReader(
        engine = engine
    )

    query = "SELECT * FROM table_descriptions"
    documents = reader.load_data(query=query)
    return documents

index = VectorStoreIndex.from_documents(load_metadata_from_sqllite())
index.storage_context.persist(table_metadata_dir / "indices/table_descriptions")
```