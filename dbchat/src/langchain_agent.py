import os

import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI


class LangchainAgent:
    def __init__(self, dataframes):
        self.dataframes = dataframes
        self.agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            self.dataframes,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def ask(self, processed_prompt):
        return agent.run(processed_prompt)


# Example usage
if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = "INSERT API KEY HERE"

    invoices = pd.read_csv("data/invoices.csv")
    customers = pd.read_csv("data/customers.csv")

    agent = LangchainAgent([invoices, customers])

    prompt = "Our sample prompts for testing, goes here"

    # Call the ask method to get the results
    results = agent.ask(prompt)

    print(results)
