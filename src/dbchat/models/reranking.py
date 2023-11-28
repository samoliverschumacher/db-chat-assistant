from dataclasses import dataclass
from typing import List, Optional

import llama_index
import llama_index.indices.vector_store.retrievers.retriever
import llama_index.objects.table_node_mapping
from llama_index import LLMPredictor, ServiceContext
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.llms import Ollama
from llama_index.objects import SQLTableSchema
from llama_index.postprocessor import LLMRerank
from llama_index.prompts.default_prompts import DEFAULT_CHOICE_SELECT_PROMPT


def parse_choice_select_answer(answer, num_choices, raise_error=False):
    """
    Given an LLM response about which documents are relevant, parses the choice 
    select answer and extracts the document numbers and relevancy scores.

    Parameters:
        answer (str): The choice select answer.
        num_choices (int): The number of choices in the answer.
        raise_error (bool, optional): Whether to raise an error if the answer cannot be parsed. Defaults to False.

    Returns:
        answer_nums (List[str]): The document numbers extracted from the answer.
        answer_relevances (List[float]): The relevancy scores extracted from the answer.
    """

    def valid_line(text):
        if not "document" in text:
            return False
        if not ": " in text:
            return False

        return True

    answer_lines = answer.split('\n')
    answer_nums = []
    answer_relevance_scores = []
    for answer_line in answer_lines:
        ans_line_lower = answer_line.lower()
        # expecting a relevancy answer in form <Document 1: Revelance score = <7>. parse to get these numbers
        try:
            if not valid_line(ans_line_lower):
                if raise_error:
                    raise ValueError(
                        "Failed to parse choice select answer"
                        "Expected answer to relevancy prompt in form;"
                        "<Document 1: revelance score = <7>.")
                continue
            doc_str, _, rel_Str = ans_line_lower.partition(':')
            val = ''.join(
                filter(lambda x: x.isnumeric(),
                       doc_str.partition('document')[2]))
            answer_nums.append(val)
            answer_relevance_scores.append(
                int(''.join(filter(lambda c: c.isnumeric(), rel_Str))[0]))
        except Exception as e:
            print(e)
            print(answer)
            raise e
    try:
        assert all([a <= 10 for a in answer_relevance_scores])

    except Exception as e:
        print(e)
        print(answer)
        raise e
    return answer_nums, answer_relevance_scores


@dataclass
class ReRankerLLMConfig:
    """A Reranker agent needs a prompt template, a response parser. Different models
    may align to the prompt differently, so a model is also paired."""

    prompt_template = DEFAULT_CHOICE_SELECT_PROMPT
    response_parser = parse_choice_select_answer
    model = 'ollama:llama2reranker'


def sql_query_engine_with_reranking(
    sql_database,
    retriever,
    service_context,
    config=ReRankerLLMConfig,
    llm_reranker: Optional[BaseLLMPredictor] = None
) -> SQLTableRetrieverQueryEngine:
    """
    Creates a SQL query engine with llm-reranking.
    
    Defaults to using an Ollama model for LLM reranking. 
     - Requires a llm configured to be deterministic.
    
    Sets the retrive function of the input retriever with additional reranking step.
    
    Uses a parsing function to parse out the ranked documents from the LLM response.
     - Parsing function depends on the prompt given to the reranker, and the LLM model's 
       ability to adhere to the prompts required answer format.
    
    Args:
        sql_database (SQLDatabase): The SQL database to be queried.
        retriever (NLSQLRetriever): The retriever used to retrieve SQL table schemas.
        service_context (ServiceContext): The service context used for LLAMA reranking.

    Returns:
        SQLTableRetrieverQueryEngine: The SQL query engine with reranking.
    """

    assert type(retriever._object_node_mapping
                ) == llama_index.objects.table_node_mapping.SQLTableNodeMapping
    assert type(
        retriever._retriever
    ) == llama_index.indices.vector_store.retrievers.retriever.VectorIndexRetriever

    # Initialise the llm, to overwrite service context
    if llm_reranker is not None:
        llm = llm_reranker
    else:
        model_source, _, model_name = config.model.partition(':')
        llm = Ollama(model=model_name)

    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    postprocessor = LLMRerank(
        top_n=3,
        service_context=service_context,
        choice_select_prompt=config.prompt_template,
        parse_choice_select_answer_fn=config.response_parser)

    # Need to postprocerss result of retrieve with the document re-ranking step.
    original_retrieve_fn = retriever._retriever.retrieve

    # Return type: same as returned function of NLSQLRetriever._load_get_tables_fn
    def retrieve_and_rerank(query_str, ) -> List[SQLTableSchema]:
        nodes_with_scores = original_retrieve_fn(query_str)

        query_bundle = QueryBundle(query_str)
        reranked_nodes = postprocessor.postprocess_nodes(
            nodes_with_scores, query_bundle)

        convert_to_schema = [
            retriever._object_node_mapping.from_node(node.node)
            for node in reranked_nodes
        ]
        return convert_to_schema

    # Set a new retrieve function
    retriever.retrieve = retrieve_and_rerank

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database, retriever, service_context=service_context)
    return query_engine
