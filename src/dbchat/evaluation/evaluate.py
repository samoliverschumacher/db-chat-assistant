from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Union

import yaml
from langchain.chat_models import ChatOpenAI
from llama_index.llms import LLM, Ollama
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from dbchat.evaluation.utils import load_evaluation_csv_data
from dbchat import sql_agent
from dbchat.validation import AppConfig


def run_batch_queries( queries: List[ str ],
                       config: dict,
                       retrieve_only = False ) -> OrderedDict[ str, Dict[ str, str ] ]:
    """Run a SQL agent pipeline on a list of queries using the given config."""
    # Create agent
    query_engine, retriever = sql_agent.create_agent( config, return_base_retriever = True )

    results = OrderedDict()
    for query in queries:

        if config[ 'approach' ] == 'sql_engine_w_reranking':
            tables = retriever.retrieve( query )
            table_names = [ table.table_name for table in tables ]
        else:
            raise NotImplementedError

        response = None
        if not retrieve_only:
            response = query_engine.query( query )
        result = {
            query: {
                'response': response,
                'tables': table_names
            },
        }
        results.update( result )

    return results


def evaluate_table_name_retrieval( test_data_path: Union[ Path, str ],
                                   config_path: Union[ Path, str ] ) -> List[ dict ]:
    """Counts the overlap between the expected and retrieved tables.
    """
    # test_data_path = "examples/evaluation/queries.csv"
    # config_path = "src/tests/data/inputs/cfg_3.yml"

    with open( config_path ) as f:
        cfg = yaml.safe_load( f )
    # Check the config is valid;
    AppConfig( **cfg )

    data = load_evaluation_csv_data( test_data_path, stream = True )
    data: Iterator[ List[ Dict[ str, str ] ] ]

    # Flatten out the data into a single list of rows
    flattened_data = list( chain.from_iterable( data ) )

    queries = [ row[ 'user_query' ] for row in flattened_data ]
    expected_tables = [ row[ 'tables' ].split( ',' ) for row in flattened_data ]

    batch_responses = run_batch_queries( queries, cfg, retrieve_only = True )

    results = []
    for ( query, response ), tables in zip( batch_responses.items(), expected_tables ):

        tables_in_expected = list( set( tables ) & set( response[ 'tables' ] ) )
        tables_not_in_expected = list( set( tables ) - set( response[ 'tables' ] ) )
        result = {
            'test_name': evaluate_table_name_retrieval.__name__,
            'config': cfg,
            'input_query': query,
            'expected_tables': tables,
            'tables_in_expected': tables_in_expected,
            'tables_not_in_expected': tables_not_in_expected
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
    - Only considers response and desired response, not context or query.
    """
    # test_data_path = "examples/evaluation/queries.csv"
    # config_path = "src/tests/data/inputs/cfg_3.yml"

    with open( config_path ) as f:
        cfg = yaml.safe_load( f )
    # Check the config is valid;
    AppConfig( **cfg )

    data = load_evaluation_csv_data( test_data_path, stream = True )
    data: Iterator[ List[ Dict[ str, str ] ] ]

    # Initialise the llm that will judge the response
    if cfg[ 'llm' ][ 'class' ] == "ollama":
        judge_llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        judge_llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Flatten out the data into a single list of rows
    flattened_data = list( chain.from_iterable( data ) )

    queries = [ row[ 'user_query' ] for row in flattened_data ]
    expected_responses = [ row[ 'response' ] for row in flattened_data ]

    batch_responses = run_batch_queries( queries, cfg, retrieve_only = True )

    results = []
    for ( query, response ), expected_response in zip( batch_responses.items(), expected_responses ):

        # Compose a prompt with the original query and the synthesized response for the "Judge" LLM
        prompt = SYNTHETIC_JUDGE_SYSTEM_PROMPT.format( expected_response = expected_response,
                                                       response = response )
        # Get a score for the user query from the "Judge" LLM
        evaluation = judge_llm.complete( prompt )

        result = {
            'test_name': evaluate_synthetic_judge.__name__,
            'config': cfg,
            'input_query': query,
            'expected_response': expected_response,
            'actual_response': response,
            'synthesized_judgement': evaluation
        }
        results.append( result )

    return results


CORRECTNESS_SYS_TMPL = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.
"""

CORRECTNESS_USER_TMPL = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""


def evaluate_synthetic_judge_with_query( test_data_path, config_path ):
    """Use an LLM as a judge that provides reasoning and a score (CORRECTNESS).
    - Considers query, response, and the ideal response.
    """
    # test_data_path = "examples/evaluation/queries.csv"
    # config_path = "src/tests/data/inputs/cfg_3.yml"

    with open( config_path ) as f:
        cfg = yaml.safe_load( f )
    # Check the config is valid;
    AppConfig( **cfg )

    data = load_evaluation_csv_data( test_data_path, stream = True )
    data: Iterator[ List[ Dict[ str, str ] ] ]

    eval_chat_template = ChatPromptTemplate( message_templates = [
        ChatMessage( role = MessageRole.SYSTEM, content = CORRECTNESS_SYS_TMPL ),
        ChatMessage( role = MessageRole.USER, content = CORRECTNESS_USER_TMPL ),
    ] )

    def run_correctness_eval(
        query_str: str,
        reference_answer: str,
        generated_answer: str,
        llm: LLM,
        threshold: float = 4.0,
    ) -> Dict:
        """Run correctness evaluation using a LLM."""
        fmt_messages = eval_chat_template.format_messages(
            llm = llm,
            query = query_str,
            reference_answer = reference_answer,
            generated_answer = generated_answer,
        )
        chat_response = llm.chat( fmt_messages )
        raw_output = chat_response.message.content

        # Extract from response
        score_str, reasoning_str = raw_output.split( "\n", 1 )
        score = float( score_str )
        reasoning = reasoning_str.lstrip( "\n" )

        return { "passing": score >= threshold, "score": score, "reason": reasoning}

    if cfg[ 'llm' ][ 'class' ] == "ollama":
        judge_llm = Ollama( model = cfg[ 'llm' ][ 'name' ] )
    else:
        judge_llm = ChatOpenAI( temperature = 0, model = "gpt-3.5-turbo-0613" )

    # Flatten out the data into a single list of rows
    flattened_data = list( chain.from_iterable( data ) )

    queries = [ row[ 'user_query' ] for row in flattened_data ]
    expected_responses = [ row[ 'response' ] for row in flattened_data ]

    batch_responses = run_batch_queries( queries, cfg, retrieve_only = True )

    results = []
    for ( query, response ), expected_response in zip( batch_responses.items(), expected_responses ):

        eval_results = run_correctness_eval( query,
                                             expected_response,
                                             response[ 'response' ],
                                             llm = judge_llm,
                                             threshold = 4.0 )

        result = {
            'test_name': evaluate_synthetic_judge_with_query.__name__,
            'config': cfg,
            'input_query': query,
            'expected_response': expected_response,
            'actual_response': response,
            'synthesized_judgement': eval_results
        }
        results.append( result )

    return results
