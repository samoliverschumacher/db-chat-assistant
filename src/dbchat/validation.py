from pydantic import BaseModel, Field, validator
from typing import Dict

from dbchat import ROOT_DIR


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


if __name__ == '__main__':
    config = load_config( ROOT_DIR.parent / 'tests/data/inputs/cfg_3.yml' )
