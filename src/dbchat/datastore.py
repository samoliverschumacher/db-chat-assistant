from enum import Enum, auto
from typing import Optional

import pandas as pd

from dbchat.query_generation import LLMAgent


try:
    from sqlalchemy import create_engine
except ImportError:
    print("Warning: SQLAlchemy is not installed.")
    pass


class types(Enum):
    SQL = auto()
    PANDAS_AGENT = auto()
    
    
def retrieve_from_pandas_agent(instructions: dict, model: LLMAgent) -> str:
    """Retrieves data from a langchain pandas agent."""
    df = pd.read_csv(instructions['data_location'])
    result = model.ask(instructions['text'])
    return result


def retrieve_from_sqllite(query: str, database_connection: dict) -> Optional[dict]:
    """Retrieves data from a sqllite database."""
    
    engine = create_engine(database_connection['connection_string'])
    
    results_as_dict = None
    with engine.begin() as connection:
        results = connection.execute(query)
        results_as_dict = results.mappings().all()
    
    return results_as_dict

    