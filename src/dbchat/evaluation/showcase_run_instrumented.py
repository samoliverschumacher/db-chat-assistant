"""
The core of trulens is in "instrumentation" on parts of the app. 
For examples of what you could instrument:
 - the prompt to LLM XXX, and the SQL code returned - to check 
   appropriate tables are found in the SQL query.
 - check the count of retrieved context documents that contain a keyword.
 - or end to end instrumentation: Given the input query, how appropriate is the 
   final response (one using a synthetic judge, another could be human 
   feedback via dashboard)
 - the SQL call to response, to check its latency
 
The below defines each instrumentation as a trulens_eval.Feedback, which is made up of:
 - an evaluation function
 - specifications on what parts of the end to end process it should apply 
   (using trulens_eval.Select.Record*).
"""

def _load_metrics( prompts: List[ dict ], config: dict ) -> List[ Feedback ]:
    """Creates evaluation metrics for the TruLens recorder."""

    sql_database = sql_agent.get_sql_database( config[ 'database' ][ 'path' ] )

    provider = StandAloneProvider( ground_truth_prompts = prompts,
                                   model_engine = "ollamallama2",
                                   possible_table_names = list( sql_database.metadata_obj.tables.keys() ) )

    ground_truth_collection = GroundTruthAgreement( ground_truth = prompts, provider = provider )

    # .rouge is word overlap metric - a built-in metric of trulens_eval.
    # on_input_output() comapres the app input to final output.
    f_groundtruth_rouge = Feedback( ground_truth_collection.rouge, name = "ROUGE" ).on_input_output()
    # A hand-rolled function uses another LLM to score out of 10, then parse the response to float
    f_groundtruth_agreement_measure = Feedback( provider.agreement_measure,
                                                name = "Agreement measure" ).on_input_output()
    retrieval_metrics = []
    for metric in config[ 'evaluation' ][ 'retrieval' ][ 'metrics' ].split( ',' ):
        if metric == 'jacard':
            retrieval_metrics.append(
                # This is metric that applies to internal steps of the LLM app, not 
                # comparing the input query with the final response. Using the FeedBack().on( ..., ... ) 
                # to specify which steps in the app sequence of processes.
                Feedback( provider.jacard_matching_tables ).on(
                    user_query = Select.RecordInput,
                    retrieved_tables = Select.Record.calls[ 0 ].rets[ : ].node.metadata.name ).aggregate(
                        np.sum ) )
        elif metric == 'accuracy':
            retrieval_metrics.append(
                # The definition of the places to apply the Feedback measurement is via json pathing.
                # i.e. Select.Record.calls[ 0 ].rets refers to a location.
                Feedback( provider.accuracy_matching_tables ).on(
                    user_query = Select.RecordInput, retrieved_tables = Select.Record.calls[ 0 ].rets ) )

    return [ f_groundtruth_rouge, f_groundtruth_agreement_measure, *retrieval_metrics ]



"""

This is the main function that defines what metrics will be 
applied to the app, and then run the app with a collection of 
Queries, and optionally their ideal responses.

"""
def run( prompts: List[ dict ], config: dict ):

    evaluation_metrics: List[ Feedback ] = _load_metrics( prompts, config )

    tru = Tru()
    tru.reset_database()  # if needed
    tru.run_dashboard()  # open a local streamlit app to explore

    query_engine = sql_agent.create_agent( config )

    tru_recorder = TruLlama( query_engine,
                             app_id = 'LlamaIndex_App1',
                             initial_app_loader = partial( sql_agent.create_agent, config ),
                             feedbacks = evaluation_metrics,
                             tru = tru )
    with tru_recorder as recording:
        for i, prompt in enumerate( prompts ):

            resp, record = tru_recorder.with_record( tru_recorder.query, prompt[ 'query' ] )
            print( resp )



if __name__ == '__main__':
    prompts = [
        {
            "query": "Which invoice has the highest Total?",
            "response": "Given the question 'Which invoice has the highest Total?', the answer is 404",
            "sql": "SELECT InvoiceId FROM invoices WHERE Total = (SELECT MAX(Total) FROM invoices);",
            "tables": [ "invoices" ]
        },
        {
            "query": "Of all the artist names, which is the longest?",
            "response": "Academy of St. Martin in the Fields, John Birch, Sir Neville Marriner & Sylvia McNair",
            "sql": "SELECT Name FROM artists ORDER BY LENGTH(Name) DESC LIMIT 1;",
            "tables": [ "invoices" ]
        },
    ]

    config_path = ROOT_DIR.parent.parent / "src/tests/data/inputs/cfg_3.yml"
    with open( config_path ) as f:
        config = yaml.safe_load( f )
    run( prompts, config )