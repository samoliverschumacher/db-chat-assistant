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
