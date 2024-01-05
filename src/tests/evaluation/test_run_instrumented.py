import pytest
from dbchat.evaluation.run_instrumented import StandAloneProvider


class TestStandAloneProvider:

    def test_extract_score_from_judgement( self ):
        judgement = 'Thank you for providing the question and expected answer. Based on the information provided, I would score the synthesized answer as follows:\n\nScore out of 10: 7\n\nThe synthesized answer provides a correct and relevant response to the question, but it could be improved by providing more context and supporting evidence for the claim made. For example, the answer could have provided the total amount of each invoice instead of just listing the numbers, or it could have explained why invoice #404 has the highest total among all the invoices provided.'
        expected_score = '7'
        assert StandAloneProvider.extract_score_from_judgement( judgement ) == expected_score

        judgement = 'Thank you for providing the question and expected answer. Based on the information provided, I would score the synthesized answer as follows:\n\nScore out of 10: 7\n\nThe synthesized answer provides a correct and relevant response to the question, but it could be improved by providing more context and explanations for why each invoice has a total of 2586. For example, the synthesized answer could explain that the numbers in the list add up to 2586, or that invoice #404 has the highest total among all the invoices provided. Overall, the synthesized answer provides a good starting point for further investigation and analysis of the given information.'
        expected_score = '7'
        assert StandAloneProvider.extract_score_from_judgement( judgement ) == expected_score


if __name__ == '__main__':
    pytest.main( [ __file__ ] )
