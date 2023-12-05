import csv
import itertools
import json
import sqlite3
from pathlib import Path
from typing import ( Dict, Iterator, List, Optional, Union, overload )

import csv
from typing import Iterator, Dict, Optional, Union, List, overload
from pathlib import Path
import itertools


def save_test_results( results, test_results_path ):
    """Appends a json array with the results."""
    if ( test_results_path ).exists():
        with open( test_results_path, 'r' ) as file:
            data = json.load( file )
    else:
        data = []
    data.append( results )

    with open( test_results_path, 'a' ) as f:
        json.dump( results, f )


import sqlite3
from typing import Iterator, Dict, Optional, Union, List, overload


@overload
def load_evaluation_sqlite_data( data_path: str,
                                 chunksize: Optional[ int ] = None,
                                 stream: bool = False ) -> List[ Dict[ str, str ] ]:
    ...


@overload
def load_evaluation_sqlite_data( data_path: str,
                                 chunksize: int = 1,
                                 stream: bool = True ) -> Iterator[ List[ Dict[ str, str ] ] ]:
    ...


def load_evaluation_sqlite_data(
        data_path: str,
        chunksize: Optional[ int ] = None,
        stream: bool = True ) -> Union[ Iterator[ List[ Dict[ str, str ] ] ], List[ Dict[ str, str ] ] ]:
    """
    Load evaluation data from a sqlite database (`data_path`).

    Either streams n=`chunksize` rows at a time, or loads all rows at once.
    """

    if not stream:
        # Connect to the SQLite database
        conn = sqlite3.connect( data_path )
        cursor = conn.cursor()

        # Execute a SELECT query to retrieve the rows from the database
        cursor.execute( "SELECT id, user_query, response, tables FROM evaluation_data" )

        rows = cursor.fetchall()
        result = [ {
            'id': row[ 0 ],
            'user_query': row[ 1 ],
            'response': row[ 2 ],
            'tables': row[ 3 ]
        } for row in rows ]

        # Close the database connection
        conn.close()

        return result
    else:
        return _load_evaluation_sqlite_data_generator( data_path, chunksize )


def _load_evaluation_sqlite_data_generator( data_path: str,
                                            chunksize: Optional[ int ] = None
                                          ) -> Iterator[ List[ Dict[ str, str ] ] ]:
    # Connect to the SQLite database
    conn = sqlite3.connect( data_path )
    cursor = conn.cursor()

    # Execute a SELECT query to retrieve the rows from the database
    cursor.execute( "SELECT id, user_query, response, tables FROM evaluation_data" )

    # Fetch rows in chunks
    while True:
        rows = cursor.fetchmany( chunksize )
        if not rows:
            break
        yield [ {
            'id': row[ 0 ],
            'user_query': row[ 1 ],
            'response': row[ 2 ],
            'tables': row[ 3 ]
        } for row in rows ]

    # Close the database connection
    conn.close()


@overload
def load_evaluation_csv_data( data_path: Union[ Path, str ],
                              delimiter: str = '|',
                              stream: bool = False,
                              chunksize = None ) -> List[ Dict[ str, str ] ]:
    ...


@overload
def load_evaluation_csv_data( data_path: Union[ Path, str ],
                              delimiter: str = '|',
                              stream: bool = True,
                              chunksize: Optional[ int ] = 1 ) -> Iterator[ List[ Dict[ str, str ] ] ]:
    ...


def load_evaluation_csv_data(
    data_path: Union[ Path, str ],
    delimiter: str = '|',
    stream: bool = True,
    chunksize: Optional[ int ] = 1
) -> Union[ List[ Dict[ str, str ] ], Iterator[ List[ Dict[ str, str ] ] ] ]:
    """
    Load evaluation data from a CSV file.

    Either streams n=`chunksize` rows at a time, or loads all rows at once.
    """

    def chunks():
        with open( str( data_path ), mode = 'r', newline = '' ) as file:
            reader = csv.DictReader( file, delimiter = delimiter )
            while True:
                rows = list( itertools.islice( reader, chunksize ) )
                if not rows:
                    break
                yield rows

    if not stream:
        with open( str( data_path ), mode = 'r', newline = '' ) as file:
            reader = csv.DictReader( file, delimiter = delimiter )
            return list( reader )
    else:
        return chunks()
