import pytest
from dbchat.evaluation.utils import load_evaluation_csv_data  # Replace 'your_module' with the actual module name.


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


if __name__ == '__main__':
    pytest.main( [ __file__ ] )
