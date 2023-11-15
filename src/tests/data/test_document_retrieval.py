from pathlib import Path
from dbchat.query_generation import document_getter
from dbchat.datastore import Table


def test_get_document():

    return
    with open( Path( "data/albums.csv" ), "r" ) as f:
        column_names = f.readline().replace( "\"", "" ).split( ',' )

        albums = Table( name = 'albums', csv = f.readlines(), fields = column_names )

    assert document_getter( 1 ) == albums
