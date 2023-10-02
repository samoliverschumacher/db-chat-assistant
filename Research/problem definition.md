# User Story

The problem we aim to address is the difficulty faced by non-data driven members of a company in effectively interacting with and deriving insights from data. Many employees, especially those without a strong background in data analysis or programming, struggle to access and interpret data to make informed decisions. This lack of data literacy hampers their ability to contribute meaningfully to data-driven discussions and hinders overall productivity and efficiency within the organization.

# Use Cases  

- Simple questions like;  
  - Which is the 5 biggest "x" (table field)
  - What sorts of things can I learn from the data?
  - What is something interesting in data table X?  
  - What is something interesting about semantic field A? (semantic field - not a literal field in a literal table, but a concept. i.e. info about a `user` might span multiple tables.)  
- Shallow analyses;  
  - Is the difference between group A and B statistically significant?  
  - Which customers has stopped interacting?  
  - What are some typical rows? i.e. perform clustering algorithm on the table.  
- Visualise;  
  - Something simple to present to other people later.  
- Inputs;  
  - I want a table that has columns A, B, C, and one row for each field "x".  
  - input a .csv to use in all other use cases.  
  
> *Must reframe the above as user intentions / "Job's to be done".*


## Problem Scope for V 0.1

### Success metrics
Returns correct SQL query for User input questions. For the test cases, a maximum of 2 tables is required to answer the below;

1. “What’s the biggest Y for observation X” Where X is a unique primary key, and Y is in the same table.
2. “In the last N days, which day of the week had the largest Y” where Y is a metric unique for each date, but not each day of the week.


The Problem space - guardrails for “Happy Path”;
- One Schema in DB only
- Tables are fully defined (primary keys, foreign keys, unique fields, data types.)
- SQLlite (or the language LLM is best suited for)
