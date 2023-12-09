import json
from typing import List
import yaml

from dbchat.evaluation.evaluate import evaluate_synthetic_judge, evaluate_synthetic_judge_with_query, evaluate_table_name_retrieval
from dbchat import ROOT_DIR

config_path = ROOT_DIR.parent / "tests/data/inputs/cfg_3.yml"
with open( config_path ) as f:
    config = yaml.safe_load( f )
print( yaml.dump( config ) )

pipeline_results: List[ dict ] = []

# For test data with only user query & expected tables;
test_data_path_no_responses = ROOT_DIR.parent.parent / "examples/evaluation/queries_no_response.csv"
if test_data_path_no_responses.exists():
    eval_funcs = [ evaluate_table_name_retrieval ]
    for f in eval_funcs:
        r = f( test_data_path_no_responses, config_path )
        pipeline_results.extend( r )

# For test data with user query, and a target desired response;
test_data_path_with_responses = ROOT_DIR.parent.parent / "examples/evaluation/queries_with_response.csv"
eval_funcs = [ evaluate_synthetic_judge_with_query, evaluate_synthetic_judge, evaluate_table_name_retrieval ]
if test_data_path_with_responses.exists():
    for f in eval_funcs:
        r = f( test_data_path_with_responses, config_path )
        pipeline_results.extend( r )

try:
    print( json.dumps( pipeline_results, indent = 2 ) )
except Exception as e:
    print( e )
    print( pipeline_results )

try:

    with open( 'pipeline_results.json', 'w' ) as f:
        json.dump( pipeline_results, f )

except Exception as e:
    print( e )
    print( pipeline_results )
