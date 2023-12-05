from pathlib import Path
from typing import List, Union
import yaml
from langchain.chat_models import ChatOpenAI
from llama_index.llms import Ollama

from dbchat.evaluation.utils import load_evaluation_csv_data
from dbchat.sql_agent import create_agent


def evaluate_table_name_retrieval( test_data_path: Union[ Path, str ],
                                   config_path: Union[ Path, str ] ) -> List[ dict ]:
    """
    test_data_path = "examples/evaluation/queries.csv"
    config_path = "src/tests/data/inputs/cfg_3.yml"
    """

    with open( config_path ) as f:
        cfg = yaml.safe_load( f )

    data = load_evaluation_csv_data( test_data_path, stream = True )
    # Embed documents, and create a retriever.
    if cfg[ 'approach' ] == 'sql_engine_w_reranking':
        _, retriever = create_agent( cfg, return_base_retriever = True )
    else:
        raise NotImplementedError( f"Approach {cfg[ 'approach' ]} not implemented" )

    results = []
    for batch in data:
        for row in batch:
            input_query, expected_tables = row[ 'user_query' ], row[ 'tables' ].split( ',' )

            # Retrieve
            if cfg[ 'approach' ] == 'sql_engine_w_reranking':
                tables = retriever.retrieve( input_query )
                table_names = [ table.table_name for table in tables ]
            else:
                raise NotImplementedError

            num_tables_in_expected = set( expected_tables ) & set( table_names )
            num_tables_not_in_expected = set( expected_tables ) - set( table_names )

            result = {
                'test_name': evaluate_table_name_retrieval.__name__,
                'config': cfg,
                'input_query': input_query,
                'expected_tables': expected_tables,
                'num_tables_in_expected': num_tables_in_expected,
                'num_tables_not_in_expected': num_tables_not_in_expected
            }
            results.append( result )

    return results


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


def evaluate_synthetic_judge( test_data_path, config_path ):
    """Use an LLM as a judge that provides reasoning and a score out of 10.

    test_data_path = "examples/evaluation/queries.csv"
    config_path = "src/tests/data/inputs/cfg_3.yml"
    """

    with open( config_path ) as f:
        cfg = yaml.safe_load( f )

    data = load_evaluation_csv_data( test_data_path, stream = True )

    query_engine = create_agent( cfg )

    results = []
    for batch in data:
        for row in batch:
            if len( row[ 'response' ] ) == 0:
                continue

            input_query, expected_response = row[ 'user_query' ], row[ 'response' ]

            response = query_engine.query( input_query )

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
            result = {
                'test_name': evaluate_synthetic_judge.__name__,
                'config': cfg,
                'input_query': input_query,
                'expected_response': expected_response,
                'actual_response': response,
                'synthesized_judgement': evaluation
            }
            results.append( result )
    return results
