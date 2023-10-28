"""
Explains the returned results to natural language text (possibly also in form of tables)
using pre trained LLMs.
"""

# Input
from enum import Enum, StrEnum
from dbchat.src.query_generation import LLMAgent


original_user_query = "Who are the top performing employees this quarter?"
sql_result = [(1, "John Smith", "Apple"),
              (2, "Jane Doe", "Microsoft")]
self_assesment = False

class named_entities(StrEnum):  # Not sure if this is correct class name?  
    user_query = "User's question"

# standard constants for prompt engineering. 
# TODO: replace constants with templated strings
prompt_constants = {"join_result_rows": '\n',
                    "template_user_query_with_answer": f"For the {named_entities.user_query}: '{{user_query}}', the data found is: ```\n{{answer}}\n```",
                    "summarise_findings": "\nPlease summarise the findings.",
                    "self_assesment": f"\nGiven the {named_entities.user_query}, does the data found make sense?"}

def explain_sql_result(original_user_query: str, sql_result: list, self_assesment: bool = False) -> str:
    # Post-process SQL result.
    _sep: str = prompt_constants['join_result_rows']
    textual_result = _sep.join([f"{name: {name}, company: {company}}" for index, (name, company) in sql_result])


    # Form a LLM prompt with SQL and original user query.
    response_prompt = prompt_constants["template_user_query_with_answer"].format(user_query=original_user_query, answer=textual_result)
    response_prompt += prompt_constants["summarise_findings"]
    if self_assesment:  # ask the LLM to check its own result for inconsistencies
        response_prompt += prompt_constants["self_assesment"]


    # Generate a response from LLM
    model = LLMAgent("model_a")
    model.load()
    result = model.ask(response_prompt)


    # Post-process LLM repsonse
    return result


def explain_pandas_agent_result(original_user_query: str, pandas_langchain_result: list, self_assessment: bool = False) -> str:
    # Post-process Langchain pandas agent result.
    
    # Form a LLM prompt with pandas_langchain and original user query.
    
    # Generate a response from LLM
    
    # Post-process LLM repsonse
    raise NotImplementedError


"""
TODO: Answer the question: How can we minimise the amount of complexity in the prompt engineering process above?
There are infinte ways to craft promts, and if we can avoid diving into prompt engineering we'll have 
more time for solving the users problem. Are there existing technologies / solutions?

Challenges:
- prompt engineering for one model, might be suboptimal for another. We may have to change models a few times 
through out this project, and dont want to re-do prompt engineering templates each time.
"""