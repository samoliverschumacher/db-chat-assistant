import logging
import os
import sys

import dotenv
from llama_index.prompts import BasePromptTemplate

from dbchat import ROOT_DIR

# load logging level from .env variables
dotenv.load_dotenv( ROOT_DIR.parent.parent / '.env' )
log_level = os.getenv( "LOG_LEVEL", "INFO" )

logging.basicConfig( stream = sys.stdout, level = log_level )
logging.getLogger().addHandler( logging.FileHandler( "sql_agent.log" ) )

from typing import Any, Optional, Tuple, Union, overload

import yaml
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OllamaEmbeddings
from llama_index import ( LLMPredictor, MockEmbedding, MockLLMPredictor, ServiceContext, SQLDatabase,
                          StorageContext, VectorStoreIndex, load_index_from_storage,
                          set_global_service_context )
from llama_index.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.indices.struct_store.sql_retriever import NLSQLRetriever
from llama_index.llms import Ollama
from llama_index.objects import ( ObjectIndex, SQLTableNodeMapping, SQLTableSchema )
from llama_index.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import MetaData, Table, create_engine, inspect

from dbchat.models.reranking import sql_query_engine_with_reranking


class CustomNLSQLRetriever( NLSQLRetriever ):

    def _get_table_context( self, query_bundle: QueryBundle ) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """
        table_schema_objs = self._get_tables( query_bundle.query_str )
        context_strs = []
        if self._context_str_prefix is not None:
            context_strs = [ self._context_str_prefix ]

        self.retrieved_tables = []
        for table_schema_obj in table_schema_objs:
            self.retrieved_tables.append( table_schema_obj.table_name )
            table_info = self._sql_database.get_single_table_info( table_schema_obj.table_name )

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append( table_info )

        return "\n\n".join( context_strs )


from llama_index.objects.base import ObjectRetriever


class CustomSQLTableRetrieverQueryEngine( SQLTableRetrieverQueryEngine ):

    def __init__( self, sql_database: SQLDatabase, table_retriever: ObjectRetriever[ SQLTableSchema ],
                  **kwargs: Any ) -> None:

        super().__init__( sql_database = sql_database, table_retriever = table_retriever, **kwargs )

        self._sql_retriever = CustomNLSQLRetriever(
            sql_database,
            **kwargs,
        )


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


def _embedding_dir_name_from_config( config: dict ):
    model_class = config[ 'index' ][ 'class' ]
    model_name = config[ 'index' ][ 'name' ]
    embedding_dir_name = f"{model_class}_{model_name}"
    persist_dir = ROOT_DIR.parent.parent / config[ 'index' ][ 'persist_dir' ] / embedding_dir_name
    return persist_dir


def _load_object_index( config: dict, sql_database: SQLDatabase ) -> ObjectIndex:
    """Used to load object index instead of the index created in `get_retriever()`"""
    table_node_mapping = SQLTableNodeMapping( sql_database )

    # rebuild storage context
    persist_dir = _embedding_dir_name_from_config( config )

    # load index
    index = ObjectIndex.from_persist_dir( persist_dir, table_node_mapping )
    return index


def _save_object_index( obj_index: ObjectIndex, table_node_mapping, config: dict ):
    """Used to save object index created in `get_retriever()`"""
    persist_dir = _embedding_dir_name_from_config( config )
    obj_index.persist( persist_dir = persist_dir, object_node_mapping = table_node_mapping )


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
    if config[ 'index' ][ 'load_embeddings' ]:
        obj_index = _load_object_index( config, sql_database )
    else:
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
    retriever = obj_index.as_retriever( **config[ 'index' ][ 'retriever_kwargs' ] )
    return retriever


def create_agent(
        config,
        debug: bool = False,
        return_base_retriever: bool = False ) -> Tuple[ SQLTableRetrieverQueryEngine, ObjectRetriever ]:
    """
    Initializes an agent by setting up the database connection (llamaindex , loading
    an embedding model, and constructing a query engine with optional
    reranking. It can optionally return the base retriever alongside
    the query engine.

    Parameters:
    - config: Configuration settings for the agent.
    - debug: Boolean flag to enable debugging features.
    - return_base_retriever: Boolean flag to return the base retriever.

    Returns:
    - If return_base_retriever is True, returns a tuple containing the
      SQLTableRetrieverQueryEngine and the base ObjectRetriever,
      otherwise just the SQLTableRetrieverQueryEngine.
    """

    db_path = config[ 'database' ][ 'path' ]

    # Initialise the encoder with a deterministic model
    if config[ 'index' ][ 'class' ] == "ollama":
        embedding_model = OllamaEmbeddings( model = config[ 'index' ][ 'name' ] )
    elif config[ 'index' ][ 'class' ] == "openai":
        embedding_model = ChatOpenAI( temperature = 0, model = config[ 'index' ][ 'name' ] )
    elif config[ 'index' ][ 'class' ] == "fake":  # For testing purposes
        embedding_model = MockEmbedding( embed_dim = config[ 'index' ][ 'embed_dim' ] )
    else:
        raise ValueError( f"Invalid index class: {config[ 'index' ][ 'class' ]}" )

    # Initialise the llm for querying
    if config[ 'llm' ][ 'name' ] == "fake":  # For testing purposes
        llm_predictor = MockLLMPredictor( max_tokens = 20 )
    else:
        llm = Ollama( model = config[ 'llm' ][ 'name' ] )
        llm_predictor = LLMPredictor( llm = llm )

    if debug:
        from llama_index.callbacks import CallbackManager, LlamaDebugHandler

        # Access the logging
        llama_debug = LlamaDebugHandler( print_trace_on_end = True )
        callback_manager = CallbackManager( [ llama_debug ] )

        service_context = ServiceContext.from_defaults( embed_model = embedding_model,
                                                        llm_predictor = llm_predictor,
                                                        callback_manager = callback_manager )
    else:
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
    # This retriever will record the table names retrieved each time a query is run.
    # query_engine._sql_retriever = CustomNLSQLRetriever( sql_database, service_context = service_context )

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
