################################################################################################
#           TEST IF RESPONSE CONTAINS THE CONTEXT IT WAS GIVEN
################################################################################################
# Checks if the synthesized response can be deduced from the sources. If not, there is halluciniation.

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, and an instruction
# for the llm to decide if the response could be given using the context information alone.

# Call the llm with the prompt.

# Score the model - "1" if it said there was enough info, "0.5" if partially, and "0" if not.

################################################################################################
#           TEST CONTEXT RETRIEVAL
################################################################################################
# Given a query, use a LLM to decide if the context retrieved contains enough to answer the query

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, and an instruction
# for the llm to decide if the context information has enough in it to answer the users query.

# Call the llm with the prompt.

# Post-process the response, score the result as "1" if it said there was enough info, "0.5" if partially,
# and "0" if not.

################################################################################################
#           TEST IF RESPONSE & SOURCE CONTEXT ANSWERS THE QUERY
################################################################################################
#

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, the synthesized response
# and an instruction for the llm to decide if the response answers the query.

# Call the llm with the prompt.

# Score the model - "1" if it said there was enough info, "0.5" if partially, and "0" if not.

# Compose the prompt with the context (document chunks), the original query, the synthesized response
# and an instruction for the llm to decide if each of the context sources contains an answer to the query

################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - CLASSICAL NLP TECHNIQUES
################################################################################################

# Use ROGUE metric to compare synthesized answers against human crafted answers
# https://huggingface.co/spaces/evaluate-metric/rouge

################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - LLM AS JUDGE
################################################################################################
# NOTE: https://gpt-index.readthedocs.io/en/v0.6.32/examples/index_structs/struct_indices/SQLIndexDemo.html
# SQL database queries

from typing import List, Tuple
from langchain.embeddings import OllamaEmbeddings
from langchain.llms.ollama import Ollama as LangchainOllama
from llama_index.llms import Ollama
from llama_index import Document, ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex
from llama_index import download_loader
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy.exc

from dbchat import ROOT_DIR
from dbchat.langchain_agent import LangchainAgent

DATA_DIR = ROOT_DIR.parent.parent / "data"
db_path = str(DATA_DIR / "chinook.db")
test_data_path = ROOT_DIR.parent / "tests/data/inputs/end-to-end.csv"


# Load example user query
def load_example_queries(test_data_path):
    test_data = []
    with open(test_data_path) as f:
        f.readline()  # Remove header row
        for row in f.readlines():
            id, user_query, tables, comment = row.split('|')
            test_data.append((id, user_query, tables, comment))
    return test_data


test_data = load_example_queries(test_data_path)
expected_response = "Berlin made $75.24"

input_query = test_data[0][1]
expected_tables = test_data[0][2].split(',')


def load_metadata_from_sqllite(
        db_path) -> Tuple[List[Document], List[Document]]:
    DatabaseReader = download_loader("DatabaseReader")

    engine = create_engine(f"sqlite:///{db_path}")
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


# Initialise the encoder (llm)
llm_llama2 = Ollama(model="llama2")
ollama_emb = OllamaEmbeddings(model="orca-mini")  # type: ignore
ctx = ServiceContext.from_defaults(llm=llm_llama2, embed_model=ollama_emb)
set_global_service_context(ctx)

# Load the documents
documents, document_names = load_metadata_from_sqllite(db_path)
document_map = {
    doc.id_: docname.text
    for docname, doc in zip(document_names, documents)
}

# Index the documents
orcamini_index = VectorStoreIndex.from_documents(documents,
                                                 service_context=ctx)
retriever = orcamini_index.as_retriever()

# Retrieve
nodes = retriever.retrieve(input_query)

#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context
context_tables = [document_map[node.node.ref_doc_id] for node in nodes]
print(f"Context tables: {context_tables}")

# load pandas dataframes from data directory, for each context_table
frames = []
for table in context_tables:
    table_path = DATA_DIR / table
    df = pd.read_csv(f"{table_path}.csv")
    frames.append(df)

# Get the LLMs response
# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
agent = LangchainAgent(dataframes=frames,
                       llm=LangchainOllama(model="llama2"))  # type: ignore
response = agent.ask(input_query)

# Compose a prompt with the original query and the synthesized response for the "Judge" LLM
SYSTEM_PROMPT = f"""
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
print(f"{SYSTEM_PROMPT=}")

# Get a score for the user query from the "Judge" LLM
# llm = Ollama(model="orca-mini")
# evaluation = llm.complete(SYSTEM_PROMPT)
evaluation = llm_llama2.complete(SYSTEM_PROMPT)
# Save the score and the explanation of the score to file.
print(f"{evaluation=}")

################################################################################################
#           TEST SYNTHESIZED RESPONSE - MANUAL EVALUATION
################################################################################################
# For each query-synthesized response pair, give a "LIKE" or "DISLIKE"
