import subprocess
import time

import pandas as pd
import requests
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OllamaEmbeddings
from llama_index import ( LLMPredictor, ServiceContext, StorageContext, VectorStoreIndex,
                          load_index_from_storage, set_global_service_context )
from llama_index.evaluation import QueryResponseEvaluator, ResponseEvaluator
from llama_index.evaluation import faithfulness as llama_index_faithfulness
from llama_index.evaluation import relevancy as llama_index_relevancy
from llama_index.evaluation.faithfulness import DEFAULT_EVAL_TEMPLATE as faithfulness_eval_template
from llama_index.evaluation.relevancy import DEFAULT_EVAL_TEMPLATE as relevancy_eval_template
from llama_index.llms import Ollama

from dbchat import ROOT_DIR, datastore
from dbchat.evaluation.utils import load_evaluation_csv_data, save_test_results
from dbchat.sql_agent import create_agent

SYNTHETIC_JUDGE_SYSTEM_PROMPT = """
Human:
You are to rate a summarization on the scale of 0-10, with 0 being completely incorrect and 10 being
a perfect summzarization of the given text.
Explain why you give your score.
Give a bullet point list of major differences between the reference and the summary.
I will supply a reference text by denoting REF: and the summarization to compare against with SUMMARY:.

REF:
{expected_response}

SUMMARY:
{response}

Assistant:"""


def setup_module( module ):
    """Run ollama serve, to make the llms available."""
    global server
    server = subprocess.Popen( [ 'ollama', 'serve' ] )

    # Poll until the server is ready
    url = 'http://localhost:11434'  #
    for _ in range( 10 ):  # try for 10 seconds
        try:
            response = requests.get( url )
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep( 0.5 )
    else:
        raise RuntimeError( 'Server did not start in 10 seconds' )


def teardown_module( module ):
    """Terminate ollama serve."""
    server.terminate()


################################################################################################
#           TEST IF RETRIEVED DOCUMENTS ARE CORRECT
################################################################################################
# Checks if the tables retrieved are correct, by comparing them to the expected tables.


def test_table_name_retrieval():
    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent.parent / "examples/evaluation/queries.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_3.yml" ) as f:
        cfg = yaml.safe_load( f )

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db( cfg[ 'database' ][ 'path' ] )
    document_map = { doc.id_: docname.text for docname, doc in zip( document_names, documents ) }

    # id matched to row in test data
    data = filter( lambda x: int( x[ 'id' ] ) == 1, load_evaluation_csv_data( test_data_path,
                                                                              stream = True ) )
    # Embed documents, and create a retriever.
    if cfg[ 'approach' ] == 'sql_engine_w_reranking':
        _, retriever = create_agent( cfg, return_base_retriever = True )
    else:
        if cfg[ 'index' ][ 'class' ] == "ollama":
            embedding_model = OllamaEmbeddings( model = cfg[ 'index' ][ 'name' ] )
        else:
            embedding_model = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )
        ctx = ServiceContext.from_defaults( embed_model = embedding_model, llm = None )
        set_global_service_context( ctx )
        index = VectorStoreIndex.from_documents( documents, service_context = ctx )
        retriever = index.as_retriever()

    for row in data:
        input_query, expected_response, expected_tables = row[ 'user_query' ], row[ 'response' ], row[
            'tables' ].split( ',' )

        # Retrieve
        nodes = retriever.retrieve( input_query )
        context_tables = [ document_map[ node.node.ref_doc_id ] for node in nodes ]

        # for each table in context_tables, check if it is in expected_tables
        num_tables_in_expected = set( expected_tables ) & set( context_tables )
        num_tables_not_in_expected = set( expected_tables ) - set( context_tables )

        results = {
            'test_name': test_table_name_retrieval.__name__,
            'config': cfg,
            'input_query': input_query,
            'expected_response': expected_response,
            'expected_tables': expected_tables,
            'num_tables_in_expected': num_tables_in_expected,
            'num_tables_not_in_expected': num_tables_not_in_expected
        }
        save_test_results( results, test_results_path )


test_table_name_retrieval()

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
#           TEST CONTEXT RETRIEVAL: "FAITHFULNESS"
################################################################################################
# Does the below context contain enough information to answer the query?
# Information is correct, coherent, but the synthesised response isnt getting it entirely from
# the context docs.
# Measurement techniques: @parmann


def test_context_retrieval():
    return  # Not applicable to pandas agent, as the document retrieval is process is not a query-context-response pair
    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml" ) as f:
        cfg = yaml.safe_load( f )
    input_query, expected_response, expected_tables = load_evaluation_data(
        test_data_path, input_id = 1 )  # id matched to row in test data

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db( cfg[ 'database' ][ 'path' ] )
    document_map = { doc.id_: docname.text for docname, doc in zip( document_names, documents ) }

    # Embed documents, and create a retriever.
    # Initialise the encoder
    if cfg[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = cfg[ 'index' ][ 'name' ] )
    else:
        embedding_model = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )
    ctx = ServiceContext.from_defaults( embed_model = embedding_model )
    set_global_service_context( ctx )
    index = VectorStoreIndex.from_documents( documents, service_context = ctx )
    retriever = index.as_retriever()

    # Retrieve
    nodes = retriever.retrieve( input_query )
    context_tables = '\n\n'.join( [ node.node.text for node in nodes ] )
    print( f"Context tables: {context_tables}" )

    # Initialise the judge llm
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        judge_llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        judge_llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )
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
    evaluation = judge_llm.complete( SYSTEM_PROMPT )

    results = {
        'test_name': test_context_retrieval.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'synthesized_judgement': evaluation
    }
    save_test_results( results, test_results_path )
    assert "1" in evaluation, "Not enough information in context to answer query"


################################################################################################
#           TEST IF RESPONSE & SOURCE CONTEXT ANSWERS THE QUERY
################################################################################################
# Whether the response is supported by the contexts or hallucinated
# based on: https://gpt-index.readthedocs.io/en/v0.6.36/how_to/evaluation/evaluation.html#evaluation-of-the-query-response-source-context


def test_faithfulness():
    """Whether the response is supported by the contexts or hallucinated."""
    return  # Not applicable to pandas agent, as the document retrieval is process is not a query-context-response pair
    judge_prompt = faithfulness_eval_template

    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml" ) as f:
        cfg = yaml.safe_load( f )
    input_query, expected_response, expected_tables = load_evaluation_data(
        test_data_path, input_id = 1 )  # id matched to row in test data

    # Initialise the llm
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Initialise the encoder
    if cfg[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = cfg[ 'index' ][ 'name' ] )
    else:
        embedding_model = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Build service context
    llm_predictor = LLMPredictor( llm = llm )
    service_context = ServiceContext.from_defaults( llm_predictor = llm_predictor,
                                                    embed_model = embedding_model )

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db( cfg[ 'database' ][ 'path' ] )
    document_map = { doc.id_: docname.text for docname, doc in zip( document_names, documents ) }

    # Embed documents, and create a retriever.
    service_context = ServiceContext.from_defaults( embed_model = embedding_model )
    set_global_service_context( service_context )
    index = VectorStoreIndex.from_documents( documents, service_context = service_context )

    # define evaluator: whether the response is supported by the contexts or hallucinated
    evaluator = llama_index_faithfulness.FaithfulnessEvaluator( service_context = service_context )

    # query index
    query_engine = index.as_query_engine()
    response = query_engine.query( input_query )
    eval_result = evaluator.evaluate( response )
    print( str( eval_result ) )
    results = {
        'test_name': test_synthetic_judge.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'actual_response': response,
        'synthesized_judgement': eval_result,
        'judge_template': judge_prompt
    }
    save_test_results( results, test_results_path )


def test_faithfulness_per_context():
    """Whether the response is supported by the contexts or hallucinated. Get result for each Context"""
    return  # Not applicable to pandas agent, as the document retrieval is process is not a query-context-response pair
    judge_prompt = faithfulness_eval_template

    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml" ) as f:
        cfg = yaml.safe_load( f )
    input_query, expected_response, expected_tables = load_evaluation_data(
        test_data_path, input_id = 1 )  # id matched to row in test data

    # Initialise the llm
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Initialise the encoder
    if cfg[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = cfg[ 'index' ][ 'name' ] )
    else:
        embedding_model = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Build service context
    llm_predictor = LLMPredictor( llm = llm )
    service_context = ServiceContext.from_defaults( llm_predictor = llm_predictor,
                                                    embed_model = embedding_model )

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db( cfg[ 'database' ][ 'path' ] )
    document_map = { doc.id_: docname.text for docname, doc in zip( document_names, documents ) }

    # Embed documents, and create a retriever.
    service_context = ServiceContext.from_defaults( embed_model = embedding_model )
    set_global_service_context( service_context )
    index = VectorStoreIndex.from_documents( documents, service_context = service_context )

    # define evaluator: whether the response is supported by the contexts or hallucinated
    evaluator = llama_index_faithfulness.FaithfulnessEvaluator( service_context = service_context )

    # query index
    query_engine = index.as_query_engine()
    response = query_engine.query( input_query )
    eval_result = evaluator.evaluate_source_nodes( response )
    print( str( eval_result ) )
    results = {
        'test_name': test_synthetic_judge.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'actual_response': response,
        'synthesized_judgement': eval_result,
        'judge_template': judge_prompt
    }
    save_test_results( results, test_results_path )


def test_faithfulness_with_query():
    """Whether the response for the query is in line with the context. Per context"""
    return  # Not applicable to pandas agent, as the document retrieval is process is not a query-context-response pair
    judge_prompt = relevancy_eval_template

    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml" ) as f:
        cfg = yaml.safe_load( f )
    input_query, expected_response, expected_tables = load_evaluation_data(
        test_data_path, input_id = 1 )  # id matched to row in test data

    # Initialise the encoder
    if cfg[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = cfg[ 'index' ][ 'name' ] )
    else:
        embedding_model = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db( cfg[ 'database' ][ 'path' ] )
    document_map = { doc.id_: docname.text for docname, doc in zip( document_names, documents ) }

    # Run the pandas agent workflow
    response = pandas_agent.run( cfg, input_query )

    # Initialise the llm
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        judge_llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        judge_llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Build service context for evaluation
    service_context = ServiceContext.from_defaults( llm_predictor = LLMPredictor( llm = judge_llm ),
                                                    embed_model = embedding_model )

    # define evaluator: whether the response is supported by the contexts or hallucinated
    evaluator = llama_index_relevancy.RelevancyEvaluator( service_context = service_context )
    eval_result_per_source = evaluator.evaluate_source_nodes( response )
    eval_result = evaluator.evaluate( response )

    results = {
        'test_name': test_synthetic_judge.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'actual_response': response,
        'synthesized_judgement': '\n'.join( [ eval_result, eval_result_per_source ] ),
        'judge_template': judge_prompt
    }
    save_test_results( results, test_results_path )


################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - CLASSICAL NLP TECHNIQUES
################################################################################################

# Use ROGUE metric to compare synthesized answers against human crafted answers
# https://huggingface.co/spaces/evaluate-metric/rouge
# METEOR, BLEU

################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - LLM AS JUDGE
################################################################################################
# Adequacy of this method requires that the llm was trained on the task of comparing two summaries
# NOTE: https://gpt-index.readthedocs.io/en/v0.6.32/examples/index_structs/struct_indices/SQLIndexDemo.html
# SQL database queries


def test_synthetic_judge():
    """Use an LLM as a judge that provides reasoning and a score out of 10."""
    test_results_path = ROOT_DIR.parent / "test_results" / "results.json"
    test_data_path = ROOT_DIR.parent / "tests" / "data" / "inputs" / "end-to-end.csv"
    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_1.yml" ) as f:
        cfg = yaml.safe_load( f )
    input_query, expected_response, expected_tables = load_evaluation_data(
        test_data_path, input_id = 1 )  # id matched to row in test data

    # Run the pandas agent workflow
    response = pandas_agent.run( cfg, input_query )

    # Initialise the llm that will judge the response
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        judge_llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        judge_llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Compose a prompt with the original query and the synthesized response for the "Judge" LLM
    prompt = SYNTHETIC_JUDGE_SYSTEM_PROMPT.format( expected_response = expected_response,
                                                   response = response )
    # Get a score for the user query from the "Judge" LLM
    evaluation = judge_llm.complete( prompt )
    # Save the score and the explanation of the score to file.
    results = {
        'test_name': test_synthetic_judge.__name__,
        'config': cfg,
        'input_query': input_query,
        'expected_tables': expected_tables,
        'expected_response': expected_response,
        'actual_response': response,
        'synthesized_judgement': evaluation
    }
    save_test_results( results, test_results_path )


################################################################################################
#           TEST SYNTHESIZED RESPONSE - MANUAL EVALUATION
################################################################################################
# For each query-synthesized response pair, give a "LIKE" or "DISLIKE"
