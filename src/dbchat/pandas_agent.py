import logging
import os
from typing import Optional

import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OllamaEmbeddings
from langchain.llms.ollama import Ollama as LangchainOllama
from langchain.schema.language_model import BaseLanguageModel
from langchain_experimental.agents.agent_toolkits import \
    create_pandas_dataframe_agent
from llama_index import (ServiceContext, VectorStoreIndex,
                         set_global_service_context)

from dbchat import datastore

# pip install openai
# pip install langchain-experimental
# conda install -c conda-forge tabulate


class PandasLangchainAgent:

    def __init__(self,
                 dataframes,
                 llm: BaseLanguageModel = ChatOpenAI(
                     temperature=0, model="gpt-3.5-turbo-0613"),
                 logger: Optional[logging.Logger] = None,
                 logfile='output.log'):
        self.logger = logger

        self.dataframes = dataframes
        self.agent = create_pandas_dataframe_agent(
            llm=llm,
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


def pandas_langchain_agent(cfg, context_tables):
    # load pandas dataframes from data directory, for each context_table
    frames = []
    for table_name in context_tables:
        rows = datastore.retrieve_from_sqllite(f"SELECT * FROM {table_name}",
                                               cfg['database']['path'])
        frames.append(pd.DataFrame(rows))

    agent = PandasLangchainAgent(
        dataframes=frames,
        llm=LangchainOllama(model=cfg['llm']['name']))  # type: ignore
    return agent


def retrieve_tables(cfg, input_query):

    # Initialise the encoder
    if cfg['index']['class'] == "ollama":
        embedding_model = OllamaEmbeddings(model=cfg['index']['name'])
    else:
        embedding_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    # Load the documents
    documents, document_names = datastore.load_metadata_from_db(
        cfg['database']['path'])
    document_map = {
        doc.id_: docname.text
        for docname, doc in zip(document_names, documents)
    }

    # Embed documents, and create a retriever.
    ctx = ServiceContext.from_defaults(embed_model=embedding_model)
    set_global_service_context(ctx)
    index = VectorStoreIndex.from_documents(documents, service_context=ctx)
    retriever = index.as_retriever()

    # Retrieve
    nodes = retriever.retrieve(input_query)
    context_tables = [document_map[node.node.ref_doc_id] for node in nodes]
    return context_tables


def run(cfg, input_query, SYSTEM_PROMPT="{input_query}"):

    input_querys = [input_query
                    ] if not isinstance(input_query, list) else input_query

    responses = []
    for input_query in input_querys:
        context_tables = retrieve_tables(cfg, input_query)

        # Compose the prompt with the context
        agent = pandas_langchain_agent(cfg, context_tables)
        responses.append(
            agent.ask(SYSTEM_PROMPT.format(input_query=input_query)))

    if len(responses) == 1:
        single_response: str = responses[0]
        return single_response
    return responses
