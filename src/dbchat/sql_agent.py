from langchain.embeddings import OllamaEmbeddings
from llama_index import (LLMPredictor, ServiceContext, SQLDatabase,
                         VectorStoreIndex, set_global_service_context)
from llama_index.llms import Ollama
from llama_index.objects import (ObjectIndex, SQLTableNodeMapping,
                                 SQLTableSchema)
from sqlalchemy import MetaData, Table, create_engine, inspect

from dbchat import ROOT_DIR
from dbchat.models.reranking import sql_query_engine_with_reranking


def get_sql_database(db_path, kwargs={}):
    """Get the SQL database."""
    engine = create_engine(f"sqlite:///{db_path}")
    inspection = inspect(engine)
    all_table_names = inspection.get_table_names()

    metadata_obj = MetaData()

    for table_name in all_table_names:
        _ = Table(table_name, metadata_obj, autoload_with=engine)
    metadata_obj.create_all(engine)

    sql_database = SQLDatabase(engine,
                               include_tables=all_table_names,
                               **kwargs)
    return sql_database


if __name__ == '__main__':

    DATA_DIR = ROOT_DIR.parent.parent / "data"
    db_path = str(DATA_DIR / "chinook.db")

    # Initialise the encoder with a deterministic model
    embedding_model = OllamaEmbeddings(model="llama2reranker")

    # Initialise the llm for querying
    llm = Ollama(model="llama2")
    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(embed_model=embedding_model,
                                                   llm_predictor=llm_predictor)
    set_global_service_context(service_context)

    sql_database = get_sql_database(db_path)

    # Construct Object Index
    table_node_mapping = SQLTableNodeMapping(sql_database)
    context_str = ""  # BUG: retrieval in 0.9.8 requires context to be set in the metadata
    table_schema_objs = [
        (SQLTableSchema(table_name=t, context_str=context_str))
        for t in sql_database.get_usable_table_names()
    ]  # add a SQLTableSchema for each table
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    retriever = obj_index.as_retriever(similarity_top_k=4)
    # Patch the SQLTableRetrieverQueryEngine, with reranking
    query_engine = sql_query_engine_with_reranking(sql_database, retriever,
                                                   service_context)

    input_query = "How much money did Berlin make?"
    response = query_engine.query(input_query)
    print(f"{input_query=}"
          ""
          f"{response=}"
          f"{response.metadata['sql_query']}")
    retrieved_tables = query_engine.sql_retriever._get_tables(input_query)
    print(f"{retrieved_tables=}")

from llama_index.postprocessor import LLMRerank


def doc_query_fusion_retriever(obj_index):
    """Creates a fusioun retriver.
    
    A Fusion retriver blends multiple retrievers by synthesising multiple 
    similar queries to the input query, and combining their retrieval results.
    See: https://github.com/Raudaschl/rag-fusion.
    """
    from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
    from llama_index.retrievers.fusion_retriever import FUSION_MODES

    # create your retrievers
    sql_retriever = obj_index.as_retriever(similarity_top_k=2)
    bm25_retriever = BM25Retriever.from_defaults(docstore=obj_index.docstore,
                                                 similarity_top_k=2)

    # create your fusion retriever
    retriever = QueryFusionRetriever(
        [sql_retriever, bm25_retriever],
        similarity_top_k=2,
        num_queries=4,  # set this to 1 to disable query generation
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )
    return retriever
