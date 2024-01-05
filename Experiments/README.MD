# TODO

## experiments
 - [ ] Make the table meta data more, or less concise, to see the effect on performance.
    - i.e. table name descriptions have words making it less concise. "Unique identifier for the track", could be "unique identifier".

# Workflows

## Perform evaluation on a ( `user query`, `desired response pair` )

1. Write the query to a csv file with headers: `id|user_query|tables|note|response`.
2. create a valid config file (validation using class: `dbchat.validation.AppConfig`).
3. in dbchat/evaluation/run_instrumented.py use function `run`.
4. Open the trulens server to view results. (`http://172.23.248.40:8501/`)

## Use additional custom content when creating documents from the SQL tables.


## Split user query test data into `with_response` and `no_response` test data .csvs

```bash
awk -F '|' '{ if ($5 == "") print > "A.csv"; else print > "B.csv" }' original.csv
```
