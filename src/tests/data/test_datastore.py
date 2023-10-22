from dbchat.datastore import retrieve_from_pandas_agent, retrieve_from_sqllite


from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

def test_retrieve_from_sqllite():
    # Test case 1: Query returns results
    query = text("SELECT * FROM albums LIMIT 2")
    database_connection = {
        'connection_string': 'sqlite:///data/chinook.db'
    }
    expected_result = [{'AlbumId': 1, 'Title': 'For Those About To Rock We Salute You', 'ArtistId': 1}, 
                       {'AlbumId': 2, 'Title': 'Balls to the Wall', 'ArtistId': 2}]
    
    assert retrieve_from_sqllite(query, database_connection) == expected_result

    # Test case 2: Query is invalid
    query = text("SELECT * FROM non_existing_table")
    database_connection = {
        'connection_string': 'sqlite:///data/chinook.db'
    }
    expected_result = OperationalError
    assert type(retrieve_from_sqllite(query, database_connection)) == expected_result
    
test_retrieve_from_sqllite()