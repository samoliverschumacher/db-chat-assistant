from llama_index import StorageContext
from llama_index.llms import Ollama
from llama_index import StorageContext, load_index_from_storage
from langchain.embeddings import OllamaEmbeddings
from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex
from langchain.chat_models import ChatOpenAI
import pandas as pd
import yaml

from dbchat import ROOT_DIR, datastore
from dbchat.langchain_agent import pandas_langchain_agent
from tests.utils import SYNTHETIC_JUDGE_SYSTEM_PROMPT, save_test_results


def get_test_data(test_data_path, input_id):
    # Load example user querys (test data, test config)

    test_data = pd.read_csv(test_data_path, sep='|')
    test_data_expected = pd.read_csv(str(test_data_path).replace(
        'inputs', 'outputs'),
                                     sep='|')

    input_query = test_data.loc[test_data.id == input_id, 'query']
    expected_tables = test_data.loc[test_data.id == input_id,
                                    'tables'].split(',')
    expected_response = test_data_expected.loc[
        test_data_expected.index == input_id,
        'response']  # "Berlin made $75.24"
    return input_query, expected_response, expected_tables


################################################################################################
#           TEST IF RETRIEVED DOCUMENTS ARE CORRECT
################################################################################################
# Checks if the tables retrieved are correct, by comparing them to the expected tables.


def test_table_name_retrieval():
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open(ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml") as f:
        cfg = yaml.safe_load(f)
    input_query, expected_response, expected_tables = get_test_data(
        test_data_path, input_id=1)  # id matched to row in test data

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db(
        cfg['database']['path'])
    document_map = {
        doc.id_: docname.text
        for docname, doc in zip(document_names, documents)
    }

    # Embed documents, and create a retriever.
    embedding_model = OllamaEmbeddings(model=cfg['index']['embedding'])
    ctx = ServiceContext.from_defaults(embed_model=embedding_model)
    set_global_service_context(ctx)
    index = VectorStoreIndex.from_documents(documents, service_context=ctx)
    retriever = index.as_retriever()

    # Retrieve
    nodes = retriever.retrieve(input_query)
    context_tables = [document_map[node.node.ref_doc_id] for node in nodes]

    # for each table in context_tables, check if it is in expected_tables
    for table in context_tables:
        assert table in expected_tables, f"Retrieved Table '{table}' not in expected tables"


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
# Does the below context contain enough information to answer the query?


def test_context_retrieval():
    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open(ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml") as f:
        cfg = yaml.safe_load(f)
    input_query, expected_response, expected_tables = get_test_data(
        test_data_path, input_id=1)  # id matched to row in test data

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db(
        cfg['database']['path'])
    document_map = {
        doc.id_: docname.text
        for docname, doc in zip(document_names, documents)
    }

    # Initialise the encoder (llm)
    if cfg['llm']['class'] == "ollama":
        judge_llm = Ollama(model=cfg['llm']['name'])
    else:
        judge_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    # Embed documents, and create a retriever.
    embedding_model = OllamaEmbeddings(model=cfg['index']['embedding'])
    ctx = ServiceContext.from_defaults(embed_model=embedding_model)
    set_global_service_context(ctx)
    index = VectorStoreIndex.from_documents(documents, service_context=ctx)
    retriever = index.as_retriever()

    # Retrieve
    nodes = retriever.retrieve(input_query)
    context_tables = '\n\n'.join([node.node.text for node in nodes])
    print(f"Context tables: {context_tables}")

    # Compose the prompt with the context
    # TODO: replace with imports from: https://gpt-index.readthedocs.io/en/v0.6.36/how_to/evaluation/evaluation.html#evaluation-of-the-query-response-source-context
    SYSTEM_PROMPT = f"""Does the below context contain enough information to answer the query? 
    
    Return a 1 if all that is needed to answer the query is contained in the context.
    Return 0.5 if only part of the information required is in the context.
    Return 0 if there is not enough information in the context to answer the query.
    
    ## Context ##
    {context_tables}
    
    ## query ##
    {input_query}"""
    evaluation = judge_llm.complete(SYSTEM_PROMPT)

    results = {
        'test_name': test_context_retrieval.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'synthesized_judgement': evaluation
    }
    save_test_results(results, test_results_path)
    assert "1" in evaluation, "Not enough information in context to answer query"


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


def test_synthetic_judge():
    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open(ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml") as f:
        cfg = yaml.safe_load(f)
    input_query, expected_response, expected_tables = get_test_data(
        test_data_path, input_id=1)  # id matched to row in test data

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db(
        cfg['database']['path'])
    document_map = {
        doc.id_: docname.text
        for docname, doc in zip(document_names, documents)
    }

    # Embed documents, and create a retriever.
    embedding_model = OllamaEmbeddings(model=cfg['index']['embedding'])
    ctx = ServiceContext.from_defaults(embed_model=embedding_model)
    set_global_service_context(ctx)
    index = VectorStoreIndex.from_documents(documents, service_context=ctx)
    retriever = index.as_retriever()

    # Retrieve
    nodes = retriever.retrieve(input_query)
    context_tables = [document_map[node.node.ref_doc_id] for node in nodes]
    print(f"Context tables: {context_tables}")

    # Compose the prompt with the context
    SYSTEM_PROMPT = f"""{input_query}"""
    agent = pandas_langchain_agent(cfg, context_tables)
    response = agent.ask(SYSTEM_PROMPT)

    # Initialise the llm that will judge the response
    if cfg['llm']['class'] == "ollama":
        judge_llm = Ollama(model=cfg['llm']['name'])
    else:
        judge_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    # Compose a prompt with the original query and the synthesized response for the "Judge" LLM
    prompt = SYNTHETIC_JUDGE_SYSTEM_PROMPT.format(
        expected_response=expected_response, response=response)
    # Get a score for the user query from the "Judge" LLM
    evaluation = judge_llm.complete(prompt)
    # Save the score and the explanation of the score to file.
    print(f"{evaluation=}")
    results = {
        'test_name': test_synthetic_judge.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'actual_response': response,
        'synthesized_judgement': evaluation
    }
    save_test_results(results, test_results_path)


################################################################################################
#           TEST SYNTHESIZED RESPONSE - MANUAL EVALUATION
################################################################################################
# For each query-synthesized response pair, give a "LIKE" or "DISLIKE"
