from langchain.embeddings import OllamaEmbeddings
from llama_index import ( LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex,
                          set_global_service_context )
from llama_index.llms import Ollama
from llama_index.objects import ( ObjectIndex, SQLTableNodeMapping, SQLTableSchema )
from llama_index.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from sqlalchemy import MetaData, Table, create_engine, inspect
import yaml

from dbchat import ROOT_DIR
from dbchat.models.reranking import sql_query_engine_with_reranking
from langchain.chat_models import ChatOpenAI

from typing import overload, Tuple


def doc_query_fusion_retriever( obj_index ):
    """Creates a fusioun retriver.

    A Fusion retriver blends multiple retrievers by synthesising multiple
    similar queries to the input query, and combining their retrieval results.
    See: https://github.com/Raudaschl/rag-fusion.
    """
    from llama_index.retrievers import BM25Retriever, QueryFusionRetriever
    from llama_index.retrievers.fusion_retriever import FUSION_MODES

    # create your retrievers
    sql_retriever = obj_index.as_retriever( similarity_top_k = 2 )
    bm25_retriever = BM25Retriever.from_defaults( docstore = obj_index.docstore, similarity_top_k = 2 )

    # create your fusion retriever
    retriever = QueryFusionRetriever(
        [ sql_retriever, bm25_retriever ],
        similarity_top_k = 2,
        num_queries = 4,  # set this to 1 to disable query generation
        mode = FUSION_MODES.RECIPROCAL_RANK,
        use_async = True,
        verbose = True,
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )
    return retriever


def get_sql_database( db_path, kwargs = {} ):
    """Get the SQL database."""
    _, prefix, path = db_path.partition( 'sqlite:///' )
    db_path = prefix + str( ( ROOT_DIR.parent.parent / path ).resolve() )

    engine = create_engine( db_path )
    inspection = inspect( engine )
    all_table_names = inspection.get_table_names()

    metadata_obj = MetaData()

    for table_name in all_table_names:
        _ = Table( table_name, metadata_obj, autoload_with = engine )
    metadata_obj.create_all( engine )

    sql_database = SQLDatabase( engine, include_tables = all_table_names, **kwargs )
    return sql_database


from sqlalchemy import create_engine, text


def load_table_descriptions( config: dict, table_names: list ):
    """
    Expects config with structure;
    metadata:
        metadata_path: "sqlite:///data/chinook.db"
        table_name: "table_descriptions"
        document_id_like: "%-1"
    """
    # Connect to the metadata database
    db_path = config[ 'metadata_path' ]
    _, prefix, path = db_path.partition( 'sqlite:///' )
    db_path = prefix + str( ( ROOT_DIR.parent.parent / path ).resolve() )

    engine = create_engine( db_path )

    # Prepare the query
    placeholders = ",".join( [ ":param" + str( i ) for i in range( len( table_names ) ) ] )
    query = f"SELECT TABLE_NAME, DESCRIPTION FROM {config['table_name']} WHERE TABLE_NAME IN ({placeholders}) AND DOCUMENT_ID LIKE :like"

    # Prepare parameters
    params = { "param" + str( i ): table_names[ i ] for i in range( len( table_names ) ) }
    params[ "like" ] = config[ 'document_id_like' ]

    # Debugging statement
    debug_query = query
    for key, value in params.items():
        debug_query = debug_query.replace( ':' + key, "'" + str( value ) + "'" )
    print( f"Debugging Query: {debug_query}" )

    # Execute the query
    with engine.connect() as conn:
        results = conn.execute( text( query ), params )

    # Create a dictionary of results
    table_descriptions = { row[ 0 ]: row[ 1 ] for row in results }
    return table_descriptions


from llama_index.objects.base import ObjectRetriever


def get_retriever( sql_database, config: dict ) -> ObjectRetriever:
    # Construct Object Index
    table_node_mapping = SQLTableNodeMapping( sql_database )
    table_names = sql_database.get_usable_table_names()
    # Load a context string for each of the tables, by querying the table metadata
    # from the database path provided in config
    table_descriptions = load_table_descriptions( config[ 'database' ][ 'metadata' ], table_names )

    # add a SQLTableSchema for each table
    table_schema_objs = [ ( SQLTableSchema( table_name = t, context_str = table_descriptions.get( t, "" ) ) )
                          for t in table_names ]
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    retriever = obj_index.as_retriever( **config[ 'index' ][ 'retriever_kwargs' ] )
    return retriever


@overload
def create_agent( config, return_base_retriever: bool = False ) -> SQLTableRetrieverQueryEngine:
    ...


@overload
def create_agent(
        config,
        return_base_retriever: bool = True ) -> Tuple[ SQLTableRetrieverQueryEngine, ObjectRetriever ]:
    ...


def create_agent( config, return_base_retriever = False ):
    db_path = config[ 'database' ][ 'path' ]

    # Initialise the encoder with a deterministic model
    if config[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = config[ 'index' ][ 'name' ] )
    elif config[ 'index' ][ 'class' ] == "openai":
        embedding_model = ChatOpenAI( temperature = 0, model = config[ 'index' ][ 'name' ] )
    else:
        raise ValueError( f"Invalid index class: {config[ 'index' ][ 'class' ]}" )

    # Initialise the llm for querying
    llm = Ollama( model = config[ 'llm' ][ 'name' ] )
    llm_predictor = LLMPredictor( llm = llm )

    service_context = ServiceContext.from_defaults( embed_model = embedding_model,
                                                    llm_predictor = llm_predictor )
    set_global_service_context( service_context )

    sql_database = get_sql_database( db_path )
    retriever = get_retriever( sql_database, config )

    if 'reranking' in config[ 'index' ]:
        # Patch the SQLTableRetrieverQueryEngine, with reranking
        query_engine = sql_query_engine_with_reranking( sql_database, retriever, service_context,
                                                        config[ 'index' ] )
    else:
        query_engine = SQLTableRetrieverQueryEngine( sql_database,
                                                     retriever,
                                                     service_context = service_context )
    if return_base_retriever:
        return query_engine, retriever
    else:
        return query_engine


if __name__ == '__main__':

    input_query = "How much money did Berlin make?"

    with open( ROOT_DIR.parent / "tests/data/inputs/cfg_3.yml" ) as f:
        config = yaml.safe_load( f )

    query_engine = create_agent( config )

    response = query_engine.query( input_query )
    print( f"{input_query=}"
           "\n"
           f"{response.response=}"
           "\n"
           f"{response.metadata['sql_query']}" )
    retrieved_tables = query_engine.sql_retriever._get_tables( input_query )
    print( f"{retrieved_tables=}" )
