"""
This is the most critical block of the solution that is responsible
for turning the user prompt and matched data to generate a database query.
"""

from dbchat.src import datastore

from typing import Protocol


class ModelProtocol(Protocol):
    def ask(self, question: str) -> str:
        ...


class ModelA:
    def ask(self, question: str) -> str:
        # Implement call to a LLM agent
        return "answer A"


class ModelB:
    def ask(self, question: str) -> str:
        # Implement call to a LLM agent
        return "answer B"


class llm:
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def load(self) -> None:
        """Load a model ready for use"""
        self._model: ModelProtocol = ModelA() if self.model_name == "model_a" else ModelB()

    def ask(self, question: str) -> str:
        try:
            return self._model.ask(question)
        except AttributeError as e:
            print("Model not loaded. Have you called load()?")
            raise e


documents = {1: "table name: employees, fields: name, organisation, manager",
             2: "table name: assets, fields: owner, value"}


def document_getter(document_index: int) -> str:
    """
    Retrieves the natural language description of the metadata given by thre input index.
    """
    
    datastore_metadata_component = documents[document_index]
    return datastore_metadata_component

def compose_data_retrieval_prompt(users_query: str, context_documents: list[str], datastore_type: datastore.types) -> str:
    """
    Adds some context infortmation to the users original query, so the LLM can decide on 
    the instructions required to retrieve the information the user wants.
    """
    if datastore_type == datastore.types.SQL:
        datastore_plain_text = "a SQL query"
    elif datastore_type == datastore.types.PANDAS_AGENT:
        datastore_plain_text = "data retireval instructions"
    # elif datastore_type == datastore.types.DATA_RETRIEVAL_INSTRUCTIONS:
    #     datastore_plain_text = "data retireval instructions"
    else:
        raise NotImplementedError(f"Unsupported datastore type: {datastore_type}")
    
    prompt = (f"A User has a need for information. Their question is: '{users_query}'"
              f"Create {datastore_plain_text} for the users question."
              "The datastore has some components in it described below;"
              '\n'.join([f"{context}" for context in context_documents]))
    
    return prompt


if __name__ == "__main__":
    
    params = dict(self_assess_repsonse = False)
    users_query = "Which is the companies highest valued asset, and who manages it?"
    
    # Using documents indexes, Retrieve the natural language version from data store.
    context_documents = [document_getter(1), document_getter(2)]
    print(f"{context_documents=}")

    # Create a prompt from the documents as context, and the original user query.
    data_retrieval_prompt = compose_data_retrieval_prompt(users_query, context_documents, datastore_type=datastore.types.SQL)

    # Generate a response from the LLM asking for appopriate data retrieval instruction
    model = llm("model_a")
    model.load()
    data_retrieval_instructions = model.ask(question=data_retrieval_prompt)

    # Optional: Ask the LLM if "there is anything wrong with the suggested SQL query, given the context"
    # (pehaps cheaper and more efficient to simply attempt the SQL query, 
    # and ask the LLM if the DB result is a valid response given the user query)
    if params['self_assess_repsonse']:
        checking_data_retrieval_instructions = ("Can you see anything wrong with this?"
                                                f"{data_retrieval_instructions}")
        corrected_data_retrieval_instructions = model.ask(question=checking_data_retrieval_instructions)
        # If the LLM found a problem with its own SQL call, overwrite it before finishing.
        if corrected_data_retrieval_instructions != "No":
            data_retrieval_instructions = corrected_data_retrieval_instructions
        
    print("Final SQL call, or data retrieval instructions "
          f"to be made to the datastore: {data_retrieval_instructions}")
