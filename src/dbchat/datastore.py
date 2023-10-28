from enum import Enum, auto
from typing import Optional

import pandas as pd

from dbchat.src.query_generation import LLMAgent

try:
    from sqlalchemy import create_engine
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    print( "Warning: SQLAlchemy is not installed." )
    pass


class types( Enum ):
    SQL = auto()
    PANDAS_AGENT = auto()
    # DATA_RETRIEVAL_INSTRUCTIONS = auto()


def retrieve_from_pandas_agent( instructions: dict, model: LLMAgent ) -> str:
    """Retrieves data from a langchain pandas agent."""
    df = pd.read_csv( instructions[ 'data_location' ] )
    result = model.ask( instructions[ 'text' ] )
    return result


def retrieve_from_sqllite( query: str, database_connection: dict ) -> Optional[ dict ]:
    """Retrieves data from a sqllite database"""

    engine = create_engine( database_connection[ 'connection_string' ] )

    results_as_dict = None
    with engine.begin() as connection:
        results = connection.execute( query )
        results_as_dict = results.mappings().all()

    return results_as_dict


from dataclasses import dataclass
from typing import List, Union

from sqlalchemy import RowMapping, Sequence, create_engine, TextClause
from sqlalchemy.exc import OperationalError


# TODO: are these classes unecessary? Does "documents" need a set of dataclasses / types,
# or can we do without the boilerplate?
@dataclass
class Table:
    name: str
    fields: List[ str ]
    csv: List[ str ]


@dataclass
class Schema:
    tables: List[ Table ]
    name: str


# TODO: is this necessary while pandas is used throughout codebase: `pd.read_sql()`
def retrieve_from_sqllite( query: TextClause,
                           database_connection: dict ) -> Union[ Sequence[ RowMapping ], OperationalError ]:
    """Retrieves data from a sqllite database"""

    engine = create_engine( database_connection[ 'connection_string' ] )

    results_as_dict = None
    with engine.begin() as connection:

        results = connection.execute( query )
        results_as_dict = results.mappings().all()

    return results_as_dict
