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