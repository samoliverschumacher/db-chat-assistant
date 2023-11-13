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
from langchain.embeddings import OllamaEmbeddings
from langchain.llms.ollama import Ollama as LangchainOllama
from llama_index.llms import Ollama
from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex
from llama_index import download_loader
import pandas as pd
from sqlalchemy import create_engine

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

def load_metadata_from_sqllite(db_path):
    DatabaseReader = download_loader("DatabaseReader")

    engine = create_engine(f"sqlite:///{db_path}")
    reader = DatabaseReader(engine = engine) # type: ignore

    query = "SELECT DESCRIPTION FROM table_descriptions"
    document_ids = reader.load_data(query=query)
    
    query = "SELECT TABLE_NAME FROM table_descriptions"
    documents = reader.load_data(query=query)
    return documents, document_ids

documents, document_ids = load_metadata_from_sqllite(db_path)
document_map = {doc.id_: docid.text for docid, doc in zip(document_ids, documents)}

# Initialise the encoder (llm)
llm_orcamini = Ollama(model="orca-mini")
ollama_emb = OllamaEmbeddings(model="orca-mini") # type: ignore
ctx = ServiceContext.from_defaults(llm=llm_orcamini, embed_model=ollama_emb)
set_global_service_context(ctx)
orcamini_index = VectorStoreIndex.from_documents(documents, service_context=ctx)
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
print(nodes)
context_tables = [document_map[node.node.id_] for node in nodes]

# load pandas dataframes from data directory, for each context_table
frames = []
for table in context_tables:
    table_path = DATA_DIR / table
    df = pd.read_csv(table_path)
    frames.append(df)

# Get the LLMs response
agent = LangchainAgent(dataframes=frames, 
                       llm=LangchainOllama(model="orca-mini")) # type: ignore
response = agent.ask(input_query)

# Compose a prompt with the original query and the synthesized response for the "Judge" LLM
prompt_data = f"""
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

# Get a score for the user query from the "Judge" LLM
llm = Ollama(model="orca-mini")
evaluation = llm.complete(prompt_data)

# Save the score and the explanation of the score to file.
print(evaluation)


################################################################################################
#           TEST SYNTHESIZED RESPONSE - MANUAL EVALUATION
################################################################################################
# For each query-synthesized response pair, give a "LIKE" or "DISLIKE"
