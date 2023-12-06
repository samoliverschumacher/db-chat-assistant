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


"""
Caching of Evaluation Results
"""

from cachetools import cached, TTLCache
import json
from typing import Tuple, Dict, Any, Union, Optional
import hashlib

# Initialize a cache with a time-to-live and a maximum size
cache = TTLCache( maxsize = 1024, ttl = 300 )


def generate_cache_key(
        cache_key: Tuple[ str, Dict[ str, Union[ Dict[ str, dict ], Dict[ str, int ] ] ], bool ] ) -> str:
    # Convert the query and retrieve_only to a string and hash them for efficiency
    key_part = ( cache_key[ 0 ], cache_key[ 2 ] )
    key_str = json.dumps( key_part, sort_keys = True )
    return hashlib.md5( key_str.encode() ).hexdigest()


@cached( cache, key = generate_cache_key )
def from_cache( cache_key: Tuple[ str, Dict[ str, Union[ Dict[ str, dict ], Dict[ str, int ] ] ], bool ],
                cache_file: str ) -> Optional[ Dict[ str, Any ] ]:
    """Retrieve cached result from a JSON file based on the cache key."""
    try:
        with open( cache_file, 'r' ) as f:
            cache_data = json.load( f )
    except FileNotFoundError:
        return None

    # Extract the query and config from cache_key
    query, config, retrieve_only = cache_key

    # Iterate over the cached data to find a match
    for stored_key, result in cache_data.items():
        stored_query, stored_config, stored_retrieve_status = stored_key

        # Compare the query strings and the configurations using the config_matches function
        if query == stored_query and retrieve_only == stored_retrieve_status:
            if config_matches( config, stored_config ):
                # Return the result if a match is found
                return result

    # If no match is found, return None
    return None


# Placeholder for config_matches function (to be implemented)
def config_matches( config, config_in_cache ):
    """Only Some parts of a runner config need to be tested for equality;
    - index/name
    - index/retriever_kwargs/*
    - index/reranking/*
    - database/metadata/document_id_like
    - llm/name
    """
    pass


def from_json_cache( cache_key: Tuple[ str, Dict[ str, Union[ Dict[ str, dict ], Dict[ str, int ] ] ], bool ],
                     cache_filepath: str ) -> Optional[ Dict[ str, Any ] ]:
    """Retrieve cached result from a JSON file based on the cache key.

    Key's are: `cache_key = ( query, config, retrieve_only )`"""
    try:
        with open( cache_filepath, 'r' ) as f:
            cache = json.load( f )
    except FileNotFoundError:
        # If the cache file does not exist, return None
        return None

    # Iterate over the cache items and check for config match
    for key, value in cache.items():
        # Split the key back into its components (query, config, retrieve_only)
        key_query, key_retrieve_only_str = key.rsplit( '_', 1 )
        key_retrieve_only = json.loads( key_retrieve_only_str )

        # Check if the query and retrieve_only matches
        if key_query == json.dumps( cache_key[ 0 ] ) and key_retrieve_only == cache_key[ 2 ]:
            # Since config_matches function is not provided, this part is pseudo-code
            if config_matches( cache_key[ 1 ], json.loads( key_query )[ 1 ] ):
                return value

    # If no match is found, return None
    return None
