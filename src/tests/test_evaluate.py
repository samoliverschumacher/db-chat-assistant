import pytest
from unittest.mock import ANY, patch, MagicMock
from dbchat.evaluation import evaluate
from collections import namedtuple

# Mock the Table namedtuple
T = namedtuple( "Table", [ "table_name" ] )


@pytest.fixture
def mock_create_agent():
    query_engine = MagicMock()
    query_engine.query.return_value = "The answer is 42."

    retriever = MagicMock()
    retriever.retrieve.return_value = [ T( "table1" ), T( "table2" ), T( "table3" ) ]

    return query_engine, retriever


def test_run_batch_queries( mock_create_agent ):
    # Unpack the mock objects from the fixture
    query_engine, retriever = mock_create_agent

    # Patch the create_agent function to return the mock objects
    with patch( 'dbchat.evaluation.evaluate.create_agent', return_value = ( query_engine, retriever ) ):
        # Define your test parameters
        queries = [ "What is the answer to life, the universe, and everything?" ]
        config = { 'approach': 'sql_engine_w_reranking'}

        # Call the function under test
        results = evaluate.run_batch_queries( queries, config )

        # Assert the results are as expected
        expected_results = {
            queries[ 0 ]: {
                'response': "The answer is 42.",
                'tables': [ "table1", "table2", "table3" ]
            },
        }
        assert results == expected_results

        # Assert that the query engine and retriever were called correctly
        query_engine.query.assert_called_once_with( queries[ 0 ] )
        retriever.retrieve.assert_called_once_with( queries[ 0 ] )


@pytest.fixture
def mock_chat_open_ai():
    mock = MagicMock()
    mock.complete.return_value = "The document scores are: DocA: 10, DocB: 9."
    return mock


@pytest.fixture
def mock_ollama():
    mock = MagicMock()
    mock.complete.return_value = "The document scores are: DocA: 10, DocB: 9."
    return mock


@pytest.fixture
def mock_load_evaluation_csv_data_as_generator():
    # Using a generator function to simulate the function call
    def generator( test_data, stream = True ):
        yield [ { 'user_query': 'whats the time?', 'response': '7pm'} ]

    # Return a callable that, when called, returns the iterator
    return generator


@pytest.fixture
def mock_run_batch_queries():
    return { 'whats the time?': { 'response': 'It is 35pm', 'tables': [ 'table1', 'table2' ]}}


@pytest.fixture
def mock_app_config():
    return MagicMock()


@pytest.fixture
def config_yaml( tmp_path ):
    # Create a temporary YAML file with the given configuration
    config_data = """
approach: "sql_engine_w_reranking"
database:
  path: "sqlite:///data/chinook.db"
  metadata:
    metadata_path: "sqlite:///data/chinook.db"
    table_name: "table_descriptions"
    document_id_like: "%-2"
index:
  name: "llama2reranker"
  class: "ollama"
  retriever_kwargs:
    similarity_top_k: 4
  reranking:
    config_object: ReRankerLLMConfig
    reranker_kwargs:
      top_n: 3
llm:
  name: "llama2"
  class: "ollama"
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text( config_data )
    return str( config_path )  # Return the path as a string


def test_evaluate_synthetic_judge( mock_ollama, mock_chat_open_ai, mock_load_evaluation_csv_data_as_generator,
                                   mock_run_batch_queries, mock_app_config, config_yaml ):
    # with patch('dbchat.evaluation.evaluate.load_evaluation_csv_data', mock_load_evaluation_csv_data_as_generator), \
    with patch('dbchat.evaluation.evaluate.load_evaluation_csv_data') as mock_load_evaluation_csv_data, \
         patch('dbchat.evaluation.evaluate.run_batch_queries', return_value=mock_run_batch_queries), \
         patch('dbchat.evaluation.evaluate.Ollama', return_value=mock_ollama), \
         patch('dbchat.evaluation.evaluate.ChatOpenAI', return_value=mock_chat_open_ai), \
         patch('dbchat.evaluation.evaluate.AppConfig', mock_app_config):
        mock_load_evaluation_csv_data.return_value = iter( [ [ {
            'user_query': 'whats the time?',
            'response': '7pm'
        } ] ] )

        test_data_path = "test_data.csv"
        config_path = config_yaml  # Use the temporary config file

        # Call the function under test
        results = evaluate.evaluate_synthetic_judge( test_data_path, config_path )

        # Assert the results are as expected
        expected_results = [ {
            'test_name': 'evaluate_synthetic_judge',
            'config': {
                'approach': "sql_engine_w_reranking",
                'database': {
                    'path': "sqlite:///data/chinook.db",
                    'metadata': {
                        'metadata_path': "sqlite:///data/chinook.db",
                        'table_name': "table_descriptions",
                        'document_id_like': "%-2"
                    }
                },
                'index': {
                    'name': "llama2reranker",
                    'class': "ollama",
                    'retriever_kwargs': {
                        'similarity_top_k': 4
                    },
                    'reranking': {
                        'config_object': "ReRankerLLMConfig",
                        'reranker_kwargs': {
                            'top_n': 3
                        }
                    }
                },
                'llm': {
                    'name': "llama2",
                    'class': "ollama"
                },
            },
            'input_query': "whats the time?",
            'actual_response': {
                'response': 'It is 35pm',
                'tables': [ 'table1', 'table2' ]
            },
            'expected_response': "7pm",
            'synthesized_judgement': "The document scores are: DocA: 10, DocB: 9."
        } ]

        # Perform your assertions here
        assert results == expected_results


@pytest.mark.parametrize( "test_data, batch_response, expected_results", [
    (
        [  # test_data
            {'user_query': 'list albums', 'tables': 'albums'},
            {'user_query': 'show tracks', 'tables': 'tracks,artists'}
        ],
        # [  # expected_tables
        #     ['albums'],
        #     ['tracks', 'artists']
        # ],
        {  # batch_response
            'list albums': {'tables': ['albums', 'artists']},
            'show tracks': {'tables': ['tracks']}
        },
        [  # expected_results
            {
                'test_name': 'evaluate_table_name_retrieval',
                'config': ANY,
                'input_query': 'list albums',
                'expected_tables': ['albums'],
                'tables_in_expected': ['albums'],
                'tables_not_in_expected': []
            },
            {
                'test_name': 'evaluate_table_name_retrieval',
                'config': ANY,
                'input_query': 'show tracks',
                'expected_tables': ['tracks', 'artists'],
                'tables_in_expected': ['tracks'],
                'tables_not_in_expected': ['artists']
            }
        ]
    ),
])  # yapf:disable
def test_evaluate_table_name_retrieval(
        config_yaml,
        test_data,
        #    expected_tables,
        batch_response,
        expected_results ):
    with patch('dbchat.evaluation.evaluate.load_evaluation_csv_data') as mock_load_evaluation_csv_data, \
            patch('dbchat.evaluation.evaluate.run_batch_queries', return_value=batch_response), \
            patch('dbchat.evaluation.evaluate.AppConfig') as mock_app_config:

        # Setup the mock functions
        mock_load_evaluation_csv_data.return_value = iter( [ test_data ] )

        # Call the function under test
        results = evaluate.evaluate_table_name_retrieval( "test_data.csv", config_yaml )

        # Check the results
        assert len( results ) == len( expected_results )
        for result, expected in zip( results, expected_results ):
            assert result[ 'test_name' ] == expected[ 'test_name' ]
            assert result[ 'input_query' ] == expected[ 'input_query' ]
            assert set( result[ 'expected_tables' ] ) == set( expected[ 'expected_tables' ] )
            assert set( result[ 'tables_in_expected' ] ) == set( expected[ 'tables_in_expected' ] )
            assert set( result[ 'tables_not_in_expected' ] ) == set( expected[ 'tables_not_in_expected' ] )


if __name__ == '__main__':
    pytest.main( [ __file__ ] )
