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

    >>> load_evaluation_csv_data( data_path, delimiter = '|', stream = False )
    [ { 'field1': 'val1', 'field2': 'val2' }, { 'field1': 'val3', 'field2': 'val4' } ]

    As streaming data from an open 3-row csv file;
    >>> gen = load_evaluation_csv_data( data_path, delimiter = '|', stream = True, chunksize = 2 )
    >>> next( gen )
    [ { 'f': 'v1' }, { 'f': 'v1' } ]
    >>> next( gen )
    [ { 'f': 'v1' } ]

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

TODO: test and implement caching for evaluation loops
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


def config_matches( config: dict,
                    config_in_cache: dict,
                    ignore_paths: List[ str ] = [],
                    include_paths: List[ str ] = [] ) -> bool:
    """Checks if 2 configs are equivalent. Only Some parts of a config need to be tested for equality;
    - index/name
    - index/retriever_kwargs/*
    - index/reranking/*
    - database/metadata/document_id_like
    - llm/name
    """
    # Hardcoded key-paths to check
    hardcoded_paths = [
        "index/name", "index/retriever_kwargs/*", "index/reranking/*", "database/metadata/document_id_like",
        "llm/name"
    ]

    # Combine hardcoded paths with include paths
    include_paths.extend( hardcoded_paths )
    include_paths = set( include_paths )

    # find paths common to both include_paths and ignore_paths
    common_paths = include_paths.intersection( ignore_paths )
    if any( common_paths ):
        raise ValueError( f"Common paths {common_paths} found in include_paths and ignore_paths. " )

    # For each key-path, check that the values are equal in both configs
    for key_path in include_paths:
        if not compare_key_paths( config, config_in_cache, key_path ):
            return False

    return True


def compare_key_paths( config: dict, config_in_cache: dict, key_path: str ) -> bool:
    # Split the key-path into individual keys
    keys = key_path.split( '/' )

    # Iterate through the keys to access the corresponding values in the configs
    for key in keys:
        if key == '*':
            # Handle wildcard key
            continue
        if key not in config or key not in config_in_cache:
            # Key not found in one of the configs
            return False

        # Update the configs to the next level
        config = config[ key ]
        config_in_cache = config_in_cache[ key ]

    # Check if the values are equal
    if config != config_in_cache:
        return False

    return True


def from_json_cache( cache_key: Tuple[ str, Dict[ str, Union[ Dict[ str, dict ], Dict[ str, int ] ] ], bool ],
                     cache_filepath: Union[ str, Path ] ) -> Optional[ Dict[ str, Any ] ]:
    """Retrieve cached result from a JSON file based on the cache key.

    Key's are: `cache_key = ( query, config, retrieve_only )`"""
    try:
        with open( cache_filepath, 'r' ) as f:
            cache = json.load( f )
    except FileNotFoundError:
        # If the cache file does not exist, return None
        return None

    # Iterate over the cache items and check for config match
    for element in cache:
        key = list( element.keys() )[ 0 ]
        value = element[ key ]
        # Split the key back into its components (query, config, retrieve_only)
        key_query, key_retrieve_only = key[ 0 ], key[ 2 ]

        # Check if the query and retrieve_only matches
        if key_query == cache_key[ 0 ] and key_retrieve_only == cache_key[ 2 ]:
            # Since config_matches function is not provided, this part is pseudo-code
            if config_matches( cache_key[ 1 ], value ):
                return value

    # If no match is found, return None
    return None
