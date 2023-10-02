"""
This is the most critical block of the solution that is responsible
for turning the user prompt and matched data to generate a database query.
"""

class llm:
    
    @staticmethod
    def load(model):
        
        class model_a: 
            def ask(self, q: str) -> str:
                return "answer A"
        
        class model_b: 
            def ask(self, q: str) -> str:
                return "answer B" 
               
        return model_a() if model=="model_a" else model_b()


datastore = {1: "table name: employees, fields: name, organisation, manager",
             2: "table name: assets, fields: owner, value"}


def document_getter(document_index: int) -> str:
    """
    Retrieves the natural language description of the metadata given by thre input index.
    """
    
    return f"datastore metadata component {document_index}"

def form_data_retrieval_prompt(users_query: str, context_documents: list[str]) -> str:
    """
    Adds some context infortmation to the users original query, so the LLM can decide on 
    the instructions required to retrieve the information the user wants.
    """
    datastore_type = "data retireval instructions"
    datastore_type = "a SQL query"
    
    prompt = (f"A User has a need for information. Their question is: '{users_query}'"
              "Create {datastore_type} for the users question."
              "The datastore has some components in it described below;"
              '\n'.join([f"{context}" for context in context_documents]))
    
    return prompt


if __name__ == "__main__":
    
    params = dict(double_check = False)
    
    users_query = "Which is the companies highest valued asset, and who manages it?"
    
    # Using documents indexes, Retrieve the natural language version from data store.
    context_documents = [document_getter(1), document_getter(2)]
    print(f"{context_documents=}")

    # Form a prompt from the documents as context, and the original user query.
    data_retrieval_prompt = form_data_retrieval_prompt(users_query, context_documents)

    # Generate a response from the LLM asking for appopriate data retrieval instruction
    model = llm.load("model_a")
    data_retrieval_instructions = model.ask(q=data_retrieval_prompt)

    # Optional: Ask the LLM if "there is anything wrong with the suggested SQL query, given the context"
    # (pehaps cheaper and more efficient to simply attempt the SQL query, 
    # and ask the LLM if the DB result is a valid response given the user query)
    if params['double_check']:
        checking_data_retrieval_instructions = ("Can you see anything wrong with this?"
                                                f"{data_retrieval_instructions}")
        corrected_data_retrieval_instructions = model.ask(q=checking_data_retrieval_instructions)
        # If the LLM found a problem with its own SQL call, overwrite it before finishing.
        if corrected_data_retrieval_instructions != "No":
            data_retrieval_instructions = corrected_data_retrieval_instructions
        
    print("Final SQL call, or data retrieval instructions "
          f"to be made to the datastore: {data_retrieval_instructions}")
