import logging
import os
from typing import Optional

import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from dbchat.logger import GitLogger


class LangchainAgent:
    def __init__(self, dataframes, logger: Optional[logging.Logger]=GitLogger):
        if logger:
            self.logger = logger

        self.dataframes = dataframes        
        self.agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            self.dataframes,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

    def ask(self, processed_prompt):
        response = agent.run(processed_prompt)
        if self.logger:
            self.logger.log(processed_prompt)
            self.logger.log(response)
            
        return response
