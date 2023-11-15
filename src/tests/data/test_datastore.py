from dbchat.datastore import retrieve_from_sqllite

import pytest

from sqlalchemy import text
from sqlalchemy.exc import OperationalError


def test_retrieve_from_sqllite():
    # Test case 1: Query returns results
    query = text( "SELECT * FROM albums LIMIT 2" )
    database_connection = { 'connection_string': 'sqlite:///data/chinook.db'}
    expected_result = [ {
        'AlbumId': 1,
        'Title': 'For Those About To Rock We Salute You',
        'ArtistId': 1
    }, {
        'AlbumId': 2,
        'Title': 'Balls to the Wall',
        'ArtistId': 2
    } ]

    assert retrieve_from_sqllite( query, database_connection ) == expected_result

    # Test case 2: Query is invalid
    query = text( "SELECT * FROM non_existing_table" )
    database_connection = { 'connection_string': 'sqlite:///data/chinook.db'}
    expected_result = OperationalError
    with pytest.raises( expected_result ):
        retrieve_from_sqllite( query, database_connection )
