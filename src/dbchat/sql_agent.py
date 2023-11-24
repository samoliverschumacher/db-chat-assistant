# import dotenv
# dotenv.load_dotenv(ROOT_DIR.parent.parent / '.env')

from langchain.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine, )
from llama_index.llms import Ollama
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index import LLMPredictor, SQLDatabase, ServiceContext, VectorStoreIndex, set_global_service_context
from sqlalchemy import Table, create_engine
from sqlalchemy import MetaData

from dbchat import ROOT_DIR


def _known_tables(tables, sql_database, query):
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables,
    )
    query_str = query
    response = query_engine.query(query_str)
    return response


def run(cfg, input_query, SYSTEM_PROMPT="{input_query}"):

    input_querys = [input_query
                    ] if not isinstance(input_query, list) else input_query

    engine = create_engine(cfg['database']['path'])
    metadata_obj = MetaData()

    sql_database = SQLDatabase(engine)

    # set Logging to DEBUG for more detailed outputs
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        (SQLTableSchema(table_name=t.name)) for t in metadata_obj.sorted_tables
    ]  # add a SQLTableSchema for each table

    # Initialise the encoder
    if cfg['index']['class'] == "ollama":
        embedding_model = OllamaEmbeddings(model=cfg['index']['name'])
    else:
        embedding_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    # Initialise the llm
    if cfg['llm']['class'] == "ollama":
        llm = Ollama(model=cfg['llm']['name'])
    else:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    llm_predictor = LLMPredictor(llm=llm)
    ctx = ServiceContext.from_defaults(embed_model=embedding_model,
                                       llm_predictor=llm_predictor)
    set_global_service_context(ctx)

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    retriever = obj_index.as_retriever(similarity_top_k=1)

    query_engine = SQLTableRetrieverQueryEngine(sql_database,
                                                retriever,
                                                service_context=ctx)

    input_querys = [input_query
                    ] if not isinstance(input_query, list) else input_query
    responses = []
    for input_query in input_querys:
        SYSTEM_PROMPT = SYSTEM_PROMPT.format(input_query=input_query)
        responses.append(query_engine.query(SYSTEM_PROMPT))

    if len(responses) == 1:
        single_response: str = responses[0]
        return single_response
    return responses


if __name__ == '__main__':

    DATA_DIR = ROOT_DIR.parent.parent / "data"
    db_path = str(DATA_DIR / "chinook.db")

    from llama_index.indices.struct_store.sql_query import (
        SQLTableRetrieverQueryEngine, )
    from llama_index.objects import (
        SQLTableNodeMapping,
        ObjectIndex,
        SQLTableSchema,
    )
    from sqlalchemy import MetaData
    from sqlalchemy import create_engine, inspect

    from llama_index import VectorStoreIndex, SQLDatabase, LLMPredictor, set_global_service_context, ServiceContext

    from langchain.embeddings import OllamaEmbeddings
    from llama_index.llms import Ollama

    # Initialise the encoder
    embedding_model = OllamaEmbeddings(model="llama2")

    # Initialise the llm
    llm = Ollama(model="llama2")
    llm_predictor = LLMPredictor(llm=llm)

    ctx = ServiceContext.from_defaults(embed_model=embedding_model,
                                       llm_predictor=llm_predictor)
    set_global_service_context(ctx)

    engine = create_engine(f"sqlite:///{db_path}")
    inspection = inspect(engine)
    all_table_names = inspection.get_table_names()

    metadata_obj = MetaData()

    for table_name in all_table_names:
        table = Table(table_name, metadata_obj, autoload_with=engine)
    metadata_obj.create_all(engine)

    sql_database = SQLDatabase(engine, include_tables=all_table_names)

    # Construct Object Index
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [(SQLTableSchema(table_name=t))
                         for t in sql_database.get_usable_table_names()
                         ]  # add a SQLTableSchema for each table
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    retriever = obj_index.as_retriever(similarity_top_k=4)

    query_engine = SQLTableRetrieverQueryEngine(sql_database,
                                                retriever,
                                                service_context=ctx)

    input_query = "How much money did Berlin make?"
    response = query_engine.query(input_query)
    print(response)

    retrieved_tables = query_engine.sql_retriever._get_tables(input_query)
    print(f"With {input_query=}, {retrieved_tables=}")
