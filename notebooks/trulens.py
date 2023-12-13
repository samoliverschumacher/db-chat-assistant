import yaml
from trulens_eval import Feedback, Provider, Select, Tru, TruCustomApp, instrument

from dbchat import ROOT_DIR, sql_agent


class StandAlone( Provider ):

    def custom_feedback( self, my_text_field: str ) -> float:
        """
        A dummy function of text inputs to float outputs.

        Parameters:
            my_text_field (str): Text to evaluate.

        Returns:
            float: square length of the text
        """
        return 1.0 / ( 1.0 + len( my_text_field ) * len( my_text_field ) )


standalone = StandAlone()
f_custom_function = Feedback( standalone.custom_feedback ).on( my_text_field = Select.RecordOutput )


class CustomApp:

    def __init__( self, config, debug = True ):
        query_engine = sql_agent.create_agent( config, debug = debug )
        self.query_engine = query_engine

    @instrument
    def respond_to_query( self, input ):
        output = self.query_engine.query( input )
        return output


with open( ROOT_DIR.parent.parent / "src/tests/data/inputs/cfg_3.yml" ) as f:
    cfg = yaml.safe_load( f )

ca = CustomApp( cfg )

# f_lang_match, f_qa_relevance, f_qs_relevance are feedback functions
tru_recorder = TruCustomApp( ca, app_id = "Custom Application v1", feedbacks = [ f_custom_function ] )

# question = "What is the total Revenue in the city Berlin?"
question = "How much did we make in Berlin?"

# To add record metadata
with tru_recorder as recording:
    recording.record_metadata = "this is metadata for all records in this context that follow this line"
    ca.respond_to_query( question )

tru = Tru()
tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed
