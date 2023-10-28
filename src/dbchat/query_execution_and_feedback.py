"""
This blocks includes executing the generated query against the 
database and provide the feedbacks as needed and also implements 
error handling mechanisms to handle cases where the user query 
does not match any data models or contains ambiguous information.
"""

import traceback
from typing import Optional, Tuple
from dbchat.src import datastore
from dbchat.src.query_generation import LLMAgent

try:
    import sqlvalidator
except ImportError:
    print("Warning: sqlvalidator is not installed.")
    
    
class NoAttemptsRemainingError(Exception):
    """Number of times the LLM was given feedback on erronous call exceeded the limit."""


def offline_sql_validation(sql_query) -> Tuple[bool, str]:
    """Uses a python package to do basic offline checks on the sql statement."""
    sql_query = sqlvalidator.parse(sql_query)
    
    if not sql_query.is_valid():
        return sql_query.is_valid(), f"{sql_query.errors}"
    return sql_query.is_valid(), sql_query


def compose_data_retrieval_instruction(agent_response: str, 
                                       datastore_type: datastore.types = datastore.types.SQL) -> str:
    """
    Takes an LLM agent's raw response to a prompt that askes for a SQL snippet, 
    and processes it ready for use as a valid SQL query.
    """
    
    if datastore_type == datastore.types.SQL:
        # trim agents response text so it only contains the SQL query
        agent_response = agent_response[agent_response.index("SELECT"): ]
        
    elif datastore_type == datastore.types.PANDAS_AGENT:
        # No preprocessing of the LLM prompt is required, it can go directly to the pandas agent
        pass
        
    return agent_response


def iteratvely_retrieve_sql_data(sql_query: str, config: dict) -> Optional[dict]:
    """
    Retrieves SQL data iteratively using the provided SQL query and configuration.
    
    This function attempts to make the database call iteratively, allowing for multiple attempts
    in case of errors. If an error occurs during the retrieval process, the function catches
    the database call errors and asks the model to try again. The model is a language model (LLM)
    that can correct a problematic sql_query and generates a new SQL query for retrieval.
    
    Args:
        sql_query (str): The SQL query used to retrieve the data.
        config (dict): The configuration settings for the data retrieval.
    
    Returns:
        Optional[dict]: The retrieved data as a dictionary, or None if retrieval fails.
    
    Raises:
        NoAttemptsRemainingError: If the maximum number of retrieval attempts is reached without success.
    """
    
    attempts_remaining = 3
    
    data = None
    model = None
    # Attempt to make the database call iteratively. 
    # If errors, catch database call errors and ask the model to try again.
    for _ in range(attempts_remaining):
        
        try:
            data = datastore.retrieve_from_sqllite(sql_query,
                                                   config)
            return data

        except Exception as e:
            relevant_error_message = traceback.format_tb(e.__traceback__)[ -1 ]
            llm_feedback_prompt = ("An error occurred while executing the SQL query."
                                  f"Error: {relevant_error_message}")
            
            # Agent needs to be loaded to iterate on the problematic sql query
            if model is None:
                model = LLMAgent(config['model'])
                model.load()
            
            retrieval_instruction = model.ask(llm_feedback_prompt)
            sql_query = compose_data_retrieval_instruction(retrieval_instruction)

    raise NoAttemptsRemainingError()


if __name__ == '__main__':
    
    # Convert natural language response into SQL statement, and query the database
    config = {'connection_string': 'sqlite:///employees.db',
              'model': 'model_a'}
    agent_response = """
    The code below selects all rows from the employees table
    
    SELECT * FROM employees;
    """
    
    data = iteratvely_retrieve_sql_data(agent_response, config)
    # Return error messages to the user, or continue
    if data is None:
        print(f"No data returned with query:\n{agent_response}")
