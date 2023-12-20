import json
from functools import partial
from itertools import dropwhile, takewhile
from pathlib import Path

from typing import Dict, List, Optional, Sequence, Union
from unittest.mock import patch

import numpy as np
import yaml
from llama_index import MockEmbedding, MockLLMPredictor
from llama_index.llms import Ollama
from llama_index.prompts import PromptType
from pydantic import Extra, Field, ValidationError
from sklearn.metrics import hamming_loss, jaccard_score
from trulens_eval import ( Feedback, Select, Tru, TruLlama )
from trulens_eval.feedback import GroundTruthAgreement
from trulens_eval.feedback.provider.base import LLMProvider

from dbchat import ROOT_DIR, sql_agent
from dbchat.validation import ( BatchTestPrompts, GoldStandardBatchTestPrompts, GoldStandardTestPrompt,
                                TestPrompt )

judge_llm = Ollama( model = 'llama2reranker' )


class StandAloneProvider( LLMProvider ):

    ground_truth_prompts: List[ dict ]
    possible_table_names: Optional[ List[ str ] ] = Field( default = None )
    table_name_indicies: Optional[ Dict[ str, int ] ] = Field( default = None )

    def __init__( self, *args, **kwargs ):
        # TODO: why was self_kwargs required here independently of kwargs?
        self_kwargs = dict()
        self_kwargs.update( **kwargs )

        if self_kwargs[ 'possible_table_names' ] is not None:
            self_kwargs[ 'table_name_indicies' ] = {
                table_name: i for i, table_name in enumerate( self_kwargs[ 'possible_table_names' ] )
            }

        super().__init__( **self_kwargs )  # need to include pydantic.BaseModel.__init__

    def jacard_matching_tables( self, user_query, retrieved_tables ):
        """Measures the similarity between the predicted set of labels and the true set of labels.
        It is calculated as the size of the intersection divided by the size of the union of the two label sets."""
        return self.matching_tables( user_query, retrieved_tables, metric = 'jacard' )

    def accuracy_matching_tables( self, user_query, retrieved_tables ):
        """Instances where the predicted labels exactly match the true labels. This must be performed on a per-instance
        basis (with all predicted tables and all actual tables).

        Extracts the table names from the `NodeWithScore` objects.
        """
        tables = []
        for retrieval in retrieved_tables:
            tables.append( retrieval[ 'node' ][ 'metadata' ][ 'name' ] )
        return self.matching_tables( user_query, retrieved_tables = tables, metric = 'accuracy' )

    def matching_tables( self,
                         user_query: str,
                         retrieved_tables: List[ str ],
                         metric = 'jacard_similarity' ) -> float:
        """Multi label classification for a single instance. `metric`'s available: hamming_loss, jacard_similarity"""
        if self.possible_table_names is None:
            raise ValueError( 'possible_table_names must be set' )

        # get the first (and only) expected tables that matches the ground truth data
        actual_tables = [
            data[ 'tables' ] for data in self.ground_truth_prompts if data[ 'query' ] == user_query
        ][ 0 ]

        # Binary vectors to represent the tables
        # create a binary valued vector from the possible table names indicies
        # self.table_name_indicies: Dict[ str, int ]
        y_pred = [ self.table_name_indicies.get( name, 0 ) for name in retrieved_tables ]
        y_true = [ self.table_name_indicies.get( name, 0 ) for name in actual_tables ]

        if metric == 'hamming_loss':
            score = hamming_loss( y_pred = y_pred, y_true = y_true )
            return score
        elif metric == 'jacard_similarity':
            # Binary average assumes that this score is aggregate with a sum.
            score = jaccard_score( y_pred = y_pred, y_true = y_true, average = 'binary' )

            return score
        elif metric == 'accuracy':
            return float( set( y_pred ) == set( y_true ) )

    def _create_chat_completion( self,
                                 prompt: Optional[ str ] = None,
                                 messages: Optional[ Sequence[ Dict ] ] = None,
                                 **kwargs ) -> str:

        ans = judge_llm.complete( prompt )
        return ans

    def agreement_measure( self, prompt, response ):
        # TODO: use a more robust way to extract the score from the synthesized judegment
        expected_answer = [
            data[ 'response' ] for data in self.ground_truth_prompts if data[ 'query' ] == prompt
        ][ 0 ]

        template = """Your job is to score a synthesized answer against an expected answer, given a question. Your answer should only be a number between 0 and 10.

        Here is an Example:
        Expected answer:
        ```
        "Green"
        ```
        Question:
        ```
        "What color are trees?"
        ```
        Synthesized answer:
        ```
        "Brown"
        ```
        Score out of 10: 0

        Below is the question, expected answer, and synthesized answer. Give me a score;

        Question:
        ```
        {query_str}
        ```
        Expected answer:
        ```
        {expected_answer}
        ```
        Synthesized answer:
        ```
        {response}
        ```
        Score out of 10:
        """
        question = template.format( query_str = prompt,
                                    expected_answer = expected_answer,
                                    response = response )
        judgement = str( judge_llm.complete( question ) )

        def extract_score_from_judgement( judgement: str ):
            # TODO: This is not perfect. Some generated judgement might not fit this extraction logic
            lines = list( reversed( judgement.split( '\n' ) ) )
            score = None
            for line in lines:
                if 'score' in line and not any( [
                        line.startswith( digit )
                        for digit in [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]
                ] ):
                    score = ''.join(
                        takewhile( lambda c: c.isdigit(), dropwhile( lambda c: not c.isdigit(), line ) ) )
                    if score:
                        break
            return score

        score = extract_score_from_judgement( judgement )
        try:
            if float( score ) / 10 > 1:
                raise ValueError( f"score out of 10: {score}" )

            return float( score ) / 10
        except ValueError as ve:
            print( f"{judgement=} \n {score=}" )
            raise ve


def load_test_data( path: Path,
                    as_dict = True ) -> Union[ BatchTestPrompts, GoldStandardBatchTestPrompts, dict ]:
    """Load test data from json file."""
    with open( path, 'r' ) as f:
        data = json.load( f )
    # Validate the test data
    try:
        data = GoldStandardBatchTestPrompts(
            test_prompts = [ GoldStandardTestPrompt( **prompt ) for prompt in data ] )
    except ValidationError as e:
        print( e )
        print( "Test data doesnt have responses and expected tables. Loading anyway." )
        data = BatchTestPrompts( test_prompts = [ TestPrompt( **prompt ) for prompt in data ] )

    if as_dict:
        return json.loads( data.json() )[ 'test_prompts' ]
    return data


class MockedEmbedding( MockEmbedding ):
    """Useful for testing non LLM components."""

    def _get_query_embedding( self, query ) -> List[ float ]:
        fp = ROOT_DIR.parent.parent / "/src/tests/data/embeddings/query_embeddings.json"
        try:
            with open( fp ) as f:
                embs = json.load( f )
            embedding = list( filter( lambda e: e[ 'query' ] == query, embs ) )[ 0 ]
            return embedding[ 'vector' ]
        except ValueError as ve:
            print( f"query: {query} not found in {fp}" )
            raise ve

    def _get_text_embedding( self, text ) -> List[ float ]:
        fp = ROOT_DIR.parent.parent / "/src/tests/data/embeddings/text_embeddings.json"
        with open( fp ) as f:
            embs = json.load( f )
        embedding = list( filter( lambda e: e[ 'text' ] == text, embs ) )[ 0 ]
        return embedding[ 'vector' ]


class MockedLLMPredictor( MockLLMPredictor ):
    """Useful for testing non LLM components."""

    def predict( self, prompt, **prompt_args ):
        # TODO: make test data dynamically generated using `load_test_data( test_data_path )`
        if prompt.metadata[ 'prompt_type' ] == PromptType.TEXT_TO_SQL:
            if prompt == "Which invoice has the highest Total?":
                return "SELECT InvoiceId FROM invoices WHERE Total = (SELECT MAX(Total) FROM invoices);"

            elif prompt == "Of all the artist names, which is the longest?":
                return "SELECT Name FROM artists ORDER BY LENGTH(Name) DESC LIMIT 1;"

        if prompt.metadata[ 'prompt_type' ] == PromptType.SQL_RESPONSE_SYNTHESIS_V2:
            return f"Given the question: '{prompt.kwargs['query_str']}', the answer is;\n\n\t'{prompt_args['context_str']}'"

        return f"Unknown query: '{prompt}'"


def _load_metrics( prompts: List[ dict ], config: dict ) -> List[ Feedback ]:
    """Creates evaluation metrics for the TruLens recorder."""

    sql_database = sql_agent.get_sql_database( config[ 'database' ][ 'path' ] )

    provider = StandAloneProvider( ground_truth_prompts = prompts,
                                   model_engine = "ollamallama2",
                                   possible_table_names = list( sql_database.metadata_obj.tables.keys() ) )

    ground_truth_collection = GroundTruthAgreement( ground_truth = prompts, provider = provider )

    f_groundtruth_rouge = Feedback( ground_truth_collection.rouge, name = "ROUGE" ).on_input_output()
    f_groundtruth_agreement_measure = Feedback( provider.agreement_measure,
                                                name = "Agreement measure" ).on_input_output()
    retrieval_metrics = []
    for metric in config[ 'evaluation' ][ 'retrieval' ][ 'metrics' ].split( ',' ):
        if metric == 'jacard':
            retrieval_metrics.append(
                Feedback( provider.jacard_matching_tables ).on(
                    user_query = Select.RecordInput,
                    retrieved_tables = Select.Record.calls[ 0 ].rets[ : ].node.metadata.name ).aggregate(
                        np.sum ) )
        elif metric == 'accuracy':
            retrieval_metrics.append(
                Feedback( provider.accuracy_matching_tables ).on(
                    user_query = Select.RecordInput, retrieved_tables = Select.Record.calls[ 0 ].rets ) )

    return [ f_groundtruth_rouge, f_groundtruth_agreement_measure, *retrieval_metrics ]


def run_mocked( prompts: List[ dict ], config: dict ):

    evaluation_metrics = _load_metrics( prompts, config )

    tru = Tru()
    tru.reset_database()  # if needed
    tru.run_dashboard()  # open a local streamlit app to explore

    with patch( 'dbchat.sql_agent.MockLLMPredictor',
                side_effect = MockedLLMPredictor ), patch( 'dbchat.sql_agent.MockEmbedding',
                                                           side_effect = MockedEmbedding ):
        # Ensure the agent is created with the Mocked Predictor and Embedding
        config[ 'llm' ][ 'name' ] = 'fake'
        config[ 'index' ][ 'class' ] = 'fake'
        del config[ 'index' ][ 'reranking' ]
        query_engine = sql_agent.create_agent( config )

        tru_recorder = TruLlama( query_engine,
                                 app_id = 'LlamaIndex_App1',
                                 initial_app_loader = partial( sql_agent.create_agent, config ),
                                 feedbacks = evaluation_metrics,
                                 tru = tru )

        with tru_recorder as recording:
            for i, prompt in enumerate( prompts ):

                resp, record = tru_recorder.with_record( tru_recorder.query, prompt[ 'query' ] )
                print( resp )


def run( prompts: List[ dict ], config: dict ):

    evaluation_metrics = _load_metrics( prompts, config )

    tru = Tru()
    tru.reset_database()  # if needed
    tru.run_dashboard()  # open a local streamlit app to explore

    query_engine = sql_agent.create_agent( config )

    tru_recorder = TruLlama( query_engine,
                             app_id = 'LlamaIndex_App1',
                             initial_app_loader = partial( sql_agent.create_agent, config ),
                             feedbacks = evaluation_metrics,
                             tru = tru )
    with tru_recorder as recording:
        for i, prompt in enumerate( prompts ):

            resp, record = tru_recorder.with_record( tru_recorder.query, prompt[ 'query' ] )
            print( resp )


if __name__ == '__main__':
    prompts = [
        {
            "query": "Which invoice has the highest Total?",
            "response": "Given the question 'Which invoice has the highest Total?', the answer is 404",
            "sql": "SELECT InvoiceId FROM invoices WHERE Total = (SELECT MAX(Total) FROM invoices);",
            "tables": [ "invoices" ]
        },
        {
            "query":
                "Of all the artist names, which is the longest?",
            "response":
                "Academy of St. Martin in the Fields, John Birch, Sir Neville Marriner & Sylvia McNair",
            "sql":
                "SELECT Name FROM artists ORDER BY LENGTH(Name) DESC LIMIT 1;",
            "tables": [ "invoices" ]
        },
        # {
        #     "query": "How much money have we made in Berlin?",
        #     "response": "$75.24",
        #     "tables": [ "invoices" ]
        # }
    ]

    config_path = ROOT_DIR.parent.parent / "src/tests/data/inputs/cfg_3.yml"
    with open( config_path ) as f:
        config = yaml.safe_load( f )
    run_mocked( prompts, config )
