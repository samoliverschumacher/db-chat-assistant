import logging
import os
from typing import Optional

import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks import FileCallbackHandler

from dbchat.logger import GitLogger

# pip isntall openai
# pip install langchain-experimental
# conda install -c conda-forge tabulate


class LangchainAgent:

    def __init__(self,
                 dataframes,
                 llm: BaseLanguageModel,
                 logger: Optional[logging.Logger] = None,
                 logfile='output.log'):
        self.logger = logger

        self.dataframes = dataframes
        self.agent = create_pandas_dataframe_agent(
            llm=llm,  #  = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
            df=self.dataframes,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        self.agent.handle_parsing_errors = True
        # TODO: design one interface for all loggers. Or make them play nicely together.
        log_handler = FileCallbackHandler(logfile)
        self.agent.callbacks = [log_handler]

    def ask(self, processed_prompt):
        runkwargs = {"input": processed_prompt}
        response = self.agent.run(**runkwargs)
        if self.logger is not None:
            msg = (f"Asking pandas dataframe agent..."
                   f"{processed_prompt}"
                   f"{response}")
            self.logger.log(logging.INFO, msg)

        return response
