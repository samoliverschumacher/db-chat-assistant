## Simple example of CSV querying app using Langchain's pandas agent and displayed with streamlit.

[link](https://dev.to/ngonidzashe/chat-with-your-csv-visualize-your-data-with-langchain-and-streamlit-ej7)
- pandas agent implements a chain of LLM queries to deduce what python pandas code to run in order to build an answer for the user. i.e.
    - `agent.run("how many people have more than 3 siblings")`
    > Entering new AgentExecutor chain...

    ```
    Thought: I need to count the number of people with more than 3 siblings
    Action: python_repl_ast
    Action Input: df[df['SibSp'] > 3].shape[0]
    Observation: 30
    Thought: I now know the final answer
    Final Answer: 30 people have more than 3 siblings.
    ```

    > Finished chain.

    '30 people have more than 3 siblings.'


## Converting sentences into embeddings
[link](https://www.sbert.net/)

This package connects to SOTA models. It's well documented, provides an alternative to manual text processing using traditional NLP. But, it may not be free or small enough to play with locally?
- perhaps we can find smaller models, or distilled versions of it?


## Review of Todays models


## Frameworks

MLFLow: "MLflow LLM Evaluate"
- Supports many types of evaluation methods
- View wevaluation results via MLflow UI
- Prompt engineering UI 
- [Recording, and Tracking runs and artefacts](https://mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded). (MLflow on localhost, MLflow on localhost with SQLite, MLflow on localhost with Tracking Server, etc.)
