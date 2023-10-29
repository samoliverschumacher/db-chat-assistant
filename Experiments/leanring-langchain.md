Looks like the ulti dataframe simply merges dataframes before applying the query to the merged frame. This could be automated without thr LLM. It saves a call to the API, because a "Name Error" occurs when the merged dataframe has a prefix applied to field names that are duplicated;
```
Invoking: `python_repl_ast` with `{'query': "df = df1.merge(df2, on='GenreId')\ndf.groupby('Name')['Milliseconds'].max().idxmax()"}`


KeyError: 'Name'
Invoking: `python_repl_ast` with `{'query': "df = df1.merge(df2, left_on='GenreId', right_on='GenreId')\ndf.groupby('Name_x')['Milliseconds'].max().idxmax()"}`
```

The process langchain pandas agent cant do, is decide which dataframes to ingest. The document retrieval will help here. Though if the agent fails, we need a way to decide if its the document retrieval step, or the langchain agent step that fails.

