from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Tuple, List

from dbchat import ROOT_DIR

try:
    import sqlvalidator  # type: ignore
except ImportError:
    print( "Warning: sqlvalidator is not installed." )


def offline_sql_validation( sql_query ) -> Tuple[ bool, str ]:
    """Uses a python package to do basic offline checks on the sql statement."""
    sql_query = sqlvalidator.parse( sql_query )  # type: ignore

    if not sql_query.is_valid():
        return sql_query.is_valid(), f"{sql_query.errors}"
    return sql_query.is_valid(), sql_query


class DatabaseConfig( BaseModel ):
    path: str
    metadata: Dict[ str, str ]

    @validator( 'path' )
    def validate_path( cls, v ):
        if not v.startswith( 'sqlite:///' ):
            raise ValueError( 'database.path must be a sqlite path' )
        return v

    @validator( 'metadata' )
    def validate_metadata( cls, v ):
        if ( '%' not in v[ 'document_id_like' ] ) and ( '_' not in v[ 'document_id_like' ] ):
            raise ValueError(
                r'database.metadata.document_id_like must conform to wildcard after SQL "LIKE" (contains % or _)'
                f": {v['document_id_like']}" )
        return v


class RetrieverKwargsConfig( BaseModel ):
    similarity_top_k: int


class RerankerKwargsConfig( BaseModel ):
    top_n: int


class IndexRerankingConfig( BaseModel ):
    config_object: str
    reranker_kwargs: RerankerKwargsConfig


class IndexConfig( BaseModel ):
    name: str
    class_: str = Field(..., alias = 'class' )
    retriever_kwargs: RetrieverKwargsConfig
    reranking: IndexRerankingConfig


class LLMConfig( BaseModel ):
    name: str
    class_: str = Field(..., alias = 'class' )


class AppConfig( BaseModel ):
    approach: str
    database: DatabaseConfig
    index: IndexConfig
    llm: LLMConfig


import yaml
from pydantic import ValidationError


def load_config( config_file ):
    try:
        with open( config_file, 'r' ) as f:
            data = yaml.safe_load( f )
        return AppConfig( **data )
    except ValidationError as e:
        print( e )  # print( e.json() )
        return None


"""Validators for evaluation data used by TruLens logging
"""


class TestPrompt( BaseModel ):
    query: str


class GoldStandardTestPrompt( TestPrompt ):
    # Hand crafted ground truth makes this a gold standard test datapoint
    response: str
    sql: str
    tables: List[ str ]


class BatchTestPrompts( BaseModel ):
    test_prompts: List[ TestPrompt ]

    @validator( 'test_prompts' )  #, check_fields = False )
    def validate_unique_prompts( cls, test_prompts ):
        prompts = [ prompt.query for prompt in test_prompts ]
        if len( prompts ) != len( set( prompts ) ):
            raise ValueError( "Duplicate prompts found" )
        return test_prompts


class GoldStandardBatchTestPrompts( BatchTestPrompts ):
    # Hand crafted ground truth makes this a gold standard test datapoint
    test_prompts: List[ GoldStandardTestPrompt ]


if __name__ == '__main__':
    config = load_config( ROOT_DIR.parent / 'tests/data/inputs/cfg_3.yml' )
