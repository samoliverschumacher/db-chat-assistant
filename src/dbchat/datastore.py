from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple
from llama_index import Document, ServiceContext, VectorStoreIndex, download_loader, set_global_service_context
from langchain.embeddings import OllamaEmbeddings

import pandas as pd
from sqlalchemy import RowMapping, text
import sqlalchemy.exc

try:
    from sqlalchemy import create_engine
except ImportError:
    print("Warning: SQLAlchemy is not installed.")
    pass


class types(Enum):
    SQL = auto()
    PANDAS_AGENT = auto()


def retrieve_from_pandas_agent(instructions: dict, model) -> str:
    """Retrieves data from a langchain pandas agent."""
    df = pd.read_csv(instructions['data_location'])
    result = model.ask(instructions['text'])
    return result


def retrieve_from_sqllite(query: str,
                          connection_string: str) -> Sequence[RowMapping]:
    """Retrieves data from a sqllite database."""

    engine = create_engine(connection_string)

    results_as_dict = None
    with engine.begin() as connection:
        results = connection.execute(text(query))
        results_as_dict = results.mappings().all()

    return results_as_dict


def document_retriever(cfg, documents):
    embedding_model = "default"
    if cfg['index']['embedding'] == "orca-mini":
        embedding_model = OllamaEmbeddings(model="orca-mini")  # type: ignore
    elif cfg['index']['embedding'] == "llama2":
        embedding_model = OllamaEmbeddings(model="llama2")  # type: ignore

    ctx = ServiceContext.from_defaults(embed_model=embedding_model)
    set_global_service_context(ctx)

    # Index the documents
    orcamini_index = VectorStoreIndex.from_documents(documents,
                                                     service_context=ctx)
    retriever = orcamini_index.as_retriever()
    return retriever


def load_metadata_from_db(db_path) -> Tuple[List[Document], List[Document]]:
    """
    Loads table metadata from a database.

    Args:
        db_path (str): The path to the database.

    Returns:
        Tuple[List[Document], List[Document]]: A tuple containing two lists of Document objects. The first list contains the document descriptions loaded from the database, and the second list contains the document names.

    Raises:
        sqlalchemy.exc.OperationalError: If there is an error accessing the database.
    """
    DatabaseReader = download_loader("DatabaseReader")

    engine = create_engine(db_path)
    reader = DatabaseReader(engine=engine)  # type: ignore
    query = "SELECT DESCRIPTION FROM table_descriptions"
    try:
        document_descriptions = reader.load_data(query=query)
    except sqlalchemy.exc.OperationalError as e:
        if "no such table" in str(e):
            print(
                "Did you add a `table_descriptions` table to DB from metadatas? "
                "See README.md for instructions.")
        raise e

    query = "SELECT TABLE_NAME FROM table_descriptions"
    document_names = reader.load_data(query=query)
    return document_descriptions, document_names
