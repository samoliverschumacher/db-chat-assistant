import json
from unittest.mock import mock_open, patch

import pytest

from dbchat.evaluation.utils import load_evaluation_csv_data
from dbchat.evaluation.utils import from_json_cache, config_matches, compare_key_paths

# TODO: check logic in all of these tests - they were written hastily with a LLM's assistance.


@pytest.fixture
def sample_csv( tmp_path ):
    data = """name|age|city
Alice|30|New York
Bob|35|Los Angeles
Charlie|25|Chicago"""
    file_path = tmp_path / "sample.csv"
    file_path.write_text( data )
    return file_path


def test_load_evaluation_csv_data_not_streaming( sample_csv ):
    result = load_evaluation_csv_data( sample_csv, stream = False )
    assert isinstance( result, list )
    assert all( isinstance( row, dict ) for row in result )
    assert len( result ) == 3  # Assuming there are 3 records in the sample CSV


def test_load_evaluation_csv_data_streaming( sample_csv ):
    result = load_evaluation_csv_data( sample_csv, stream = True, chunksize = 1 )
    assert hasattr( result, '__iter__' )  # Check if result is iterable (like an iterator)
    chunk = next( result )
    assert isinstance( chunk, list )
    assert len( chunk ) == 1  # Check if chunksize works correctly


def test_load_evaluation_csv_data_streaming_no_chunksize( sample_csv ):
    result = load_evaluation_csv_data( sample_csv, stream = True, chunksize = None )
    # Get all chunks (should be only one chunk with all rows)
    all_chunks = list( result )
    assert len( all_chunks ) == 1  # There should be only one chunk
    assert len( all_chunks[ 0 ] ) == 3  # The chunk should contain all records


def test_load_evaluation_csv_data_wrong_delimiter( sample_csv ):
    result = load_evaluation_csv_data( sample_csv, delimiter = ',', stream = False )
    assert len( result ) == 3
    assert result[ 0 ].get( 'name|age|city' ) is not None  # The wrong delimiter should not split the columns


def test_load_evaluation_csv_data_file_not_found():
    with pytest.raises( FileNotFoundError ):
        load_evaluation_csv_data( 'nonexistent.csv', stream = False )


# Test when the cache file is not found
def test_cache_file_not_found():
    mock_cache_file = 'path/to/nonexistent_cache.json'
    input_cache_key = ( 'queryA', { 'key1': 'value1', 'key2': 'value2'}, True )

    with patch( 'builtins.open', mock_open() ) as mocked_open:
        mocked_open.side_effect = FileNotFoundError

        # with pytest.raises( FileNotFoundError ):
        result = from_json_cache( input_cache_key, mock_cache_file )
        assert result is None


@patch( 'dbchat.evaluation.utils.compare_key_paths' )
def test_config_matches( mock_compare_key_paths ):
    # Set the return value of the mock compare_key_paths function to True
    mock_compare_key_paths.return_value = True

    # Your test configs
    config = {
        "index": {
            "name": "config_index",
            "retriever_kwargs": {
                "arg1": "value1",
                "arg2": "value2"
            },
            "reranking": {
                "rank1": "value1"
            }
        },
        "database": {
            "metadata": {
                "document_id_like": "123"
            }
        },
        "llm": {
            "name": "config_llm"
        }
    }

    config_in_cache = {
        "index": {
            "name": "config_index",
            "retriever_kwargs": {
                "arg1": "value1",
                "arg2": "value2"
            },
            "reranking": {
                "rank1": "value1"
            }
        },
        "database": {
            "metadata": {
                "document_id_like": "123"
            }
        },
        "llm": {
            "name": "config_llm"
        }
    }

    # Test with the mock compare_key_paths function
    assert config_matches( config, config_in_cache, ignore_paths = [], include_paths = [] )

    # Test with a path that is both included and ignored
    with pytest.raises( ValueError ):
        config_matches( config,
                        config_in_cache,
                        ignore_paths = [ 'index/name' ],
                        include_paths = [ 'index/name' ] )


def test_compare_key_paths_matching():
    # Test data where key paths are matching
    config = { "index": { "name": "test_index", "retriever_kwargs": { "arg1": "value1",}}}

    config_in_cache = { "index": { "name": "test_index", "retriever_kwargs": { "arg1": "value1",}}}

    # The key path that we want to compare
    key_path = "index/retriever_kwargs/arg1"

    # Call the function and assert that the key paths are matching
    assert compare_key_paths( config, config_in_cache, key_path )


def test_compare_key_paths_not_matching():
    # Test data where key paths are not matching
    config = { "index": { "name": "test_index", "retriever_kwargs": { "arg1": "value1",}}}

    config_in_cache = {
        "index": {
            "name": "test_index",
            "retriever_kwargs": {
                "arg1": "value2",  # Different value than in `config`
            }
        }
    }

    # The key path that we want to compare
    key_path = "index/retriever_kwargs/arg1"

    # Call the function and assert that the key paths are not matching
    assert not compare_key_paths( config, config_in_cache, key_path )


# Test when no match is found in the cache
def test_no_match_found():
    # Mock data that does not match the input cache_key
    mock_cache = [ {
        ( 'query A', json.dumps( {
            'key1': 'value1',
            'key2': 'value2'
        } ), True ): {
            'response': 'the answer',
            'tables': [ 'table1', 'table2' ]
        }
    } ]
    mock_cache_file = 'path/to/mock_cache.json'
    input_cache_key = ( 'query B', { 'key1': 'value1', 'key2': 'value2'}, True )

    with patch( 'json.load', return_value = mock_cache ), patch( 'builtins.open', mock_open() ):
        result = from_json_cache( input_cache_key, mock_cache_file )

        assert result is None  # No match should return None


def test_match_found():
    mock_cache = [ {
        ( 'queryA', json.dumps( {
            'key1': 'value1',
            'key2': 'value2'
        } ), True ): {
            'response': 'the answer',
            'tables': [ 'table1', 'table2' ]
        }
    } ]
    mock_cache_file = 'path/to/mock_cache.json'
    input_cache_key = ( 'queryA', { 'key1': 'value1', 'key2': 'value2'}, True )
    expected_value = { 'response': 'the answer', 'tables': [ 'table1', 'table2' ]}

    with patch( 'json.load', return_value = mock_cache ), patch( 'builtins.open', mock_open() ):
        with patch( 'dbchat.evaluation.utils.config_matches', return_value = True ):
            result = from_json_cache( input_cache_key, mock_cache_file )
            assert result == expected_value


if __name__ == '__main__':
    pytest.main( [ __file__ ] )
