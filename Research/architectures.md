## LLama-index SQL Engine

`retrieve_with_metadata` prompts LLM to generate a SQL query given some table metadata. It assumes the SQL call is correct, and calls the database.
- If it errors, the node returned will be an error node. There is no retry procedure.
