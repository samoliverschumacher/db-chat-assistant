# db-chat-assistant
connects a database schema information to a LLM to give results without knowing SQL

# The Problem
- People without SQL or tableau knowledge can't make data driven decisions on their own.

# Exploring Solutions

> A quick search online shows a lot of solutions already available for this problem.

[Creating a Chatbot Based on ChatGPT for Interacting with Databases](https://www.clearpeaks.com/creating-a-chatbot-based-on-chatgpt-for-interacting-with-databases/) solves the problem.
- [LLama index](https://github.com/jerryjliu/llama_index) is a way to index datasets, and structure the representation of SQL for the LLM.
- Possibly would require manually describing each data table first?

## Possible challenges to consider
- Describing what each table in the DB is about. So the LLM can interpret a query and use those descriptions to help decide which table the query is about.
- Manually listing all the foreign keys, as a separte data source for the LLM to use as context.
- Adding in prompts that help translate ambiguous queries. i.e. "If the user asks about truck types, they mean the field cargo_type".
