For generative AI, what should our tests be focusing on?

What do we assume already works for an LLM?
- i.e. "Checking the agent can recall info from a length N context", has already been done - we dont want to test the dependencies of our project.

We test the user intent, but it's the user query passed into the system.
- User intents such as "Intent 1: Describe the answer, Intent 2: Reason about the answer", might map to a larger number of possible inputs. And those inputs might map to a larger number of expected outputs

# If not the user intent, test what you can

## Qualitative tests:
 - "I Like this response" or "I Don't like"
 - "This response gets the user closer to answering their query"

## Or Programattic tests: 

> Two types of tests;
>  - Check similarity between synthesized result, and a reference answer defined by a human.
>  - Use LLM to evaluate correctness of Synthesized response, or the steps leading to it (documents retrieved etc.).

And system performance tests - cost, time etc.

### Comparing Synthesized results with Reference result
Cross-checking with SQL queries that the query was correct. User queries would have to be simple, i.e. *"What is the largest number in the table?"*

Comparison between multiple human defined summaries, and the systems automated summary;
- Classical NLP based scoring like [ROGUE](https://huggingface.co/spaces/evaluate-metric/rouge)
- LLMs to perform the "scoring" of semantic similarity between a synthesized summary, and the human defined summary.
    - Prompt the LLM to;
        - Score a synthesized response, given a human defined "perfect" answer.
        - Require explanation of the score given (for inspection / transparency).

### LLM as a judge of RAG results

[LLM to check if retrieved context matches the Retrieval augmented Synthesized response (llamaindex)](https://gpt-index.readthedocs.io/en/v0.6.36/how_to/evaluation/evaluation.html#evaluation-of-the-response-context)

[LLM to evaluate Appropriateness of Synthesized response given query and context documents (llamaindex)](https://gpt-index.readthedocs.io/en/v0.6.36/how_to/evaluation/evaluation.html#evaluation-of-the-query-response-source-context)

[LLM to asses if the retrieved sources contain an answer to the query (llamaindex)](https://gpt-index.readthedocs.io/en/v0.6.36/how_to/evaluation/evaluation.html#id3)

Test creation: Given source document(s), LLM generates possible questions for the RAG system to be evaluated on.

References;
- https://pub.towardsai.net/how-to-evaluate-the-quality-of-llm-based-chatbots-ae62abe43068
- [Ideas for evaluation methods](https://towardsdatascience.com/how-to-measure-the-success-of-your-rag-based-llm-system-874a232b27eb)
- [LLM as judge for RAG results](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)

## Log: Concenrns on evaluation

### 1/11/23
A user query used to match to a vector of documents (metadata of a table description), might not be the right solution to the problem.

Need to mimic the way human does it.
    "which df is the query related to?"
 - We look at table names, and table column names.
