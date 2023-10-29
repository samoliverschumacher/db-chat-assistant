from itertools import filterfalse
import os
from datetime import datetime
from pathlib import Path
import subprocess
import logging

import pandas as pd
from dbchat.langchain_agent import LangchainAgent

from dbchat.logger import GitLogger as Logger
from dbchat import ROOT_DIR

        
if __name__ == '__main__':
    
    log_dir = Path("Experiments", "logs", "pandas_agent")
    logger = Logger(log_dir)
    
    os.environ["OPENAI_API_KEY"] = "api-key-goes-here"

    test_data_path = Path(__file__).parent / "../src/tests/data/inputs/end-to-end.csv"
    test_results_path = Path(str(test_data_path).replace('inputs', 'outputs'))
    db_path = str(Path(ROOT_DIR.parent.parent, "data/chinook.db").resolve())
    
    test_data = []
    with open(test_data_path) as f:
        f.readline()  # Remove header row
        for row in f.readlines():
            id, user_query, tables, comment = row.split('|')
            test_data.append((id, user_query, tables, comment))
            
    with open(test_results_path) as f:
        results_header = f.readline()
        results_header.index('query_id')
        
    single_table_data = list(filter(lambda r: len(r[2].split(','))==1, test_data ))
    multi_table_data = list(filterfalse(lambda r: len(r[2].split(','))==1, test_data ))
    
    test_single = False
    if test_single:
        outputs = []
        for test_id, user_query, table, _ in single_table_data:
            
            df = pd.read_sql(f"SELECT * FROM {table}", con=f"sqlite:///{db_path}")
            agent = LangchainAgent(df, logger=logger)

            # Call the ask method to get the results
            results = agent.ask(user_query)
            print(results)
            
            outputs.append((test_id, user_query, results))
            
    test_multi = True
    if test_multi:
        outputs = []
        test_id, user_query, tables, _ = multi_table_data[1]
        employees,customers,invoices,invoice_items = tables.split(',')
        
        df = pd.read_sql(f"SELECT * FROM {employees}", con=f"sqlite:///{db_path}")
        df1 = pd.read_sql(f"SELECT * FROM {customers}", con=f"sqlite:///{db_path}")
        df2 = pd.read_sql(f"SELECT * FROM {invoices}", con=f"sqlite:///{db_path}")
        df3 = pd.read_sql(f"SELECT * FROM {invoice_items}", con=f"sqlite:///{db_path}")
        
        agent = LangchainAgent([df, df1, df2, df3], logger=logger)

        # Call the ask method to get the results
        results = agent.ask(user_query)
        print(results)
        
        outputs.append((test_id, user_query, results))
    
    with open(test_results_path, 'a') as f:
        for test_id, user_query, results in outputs:
            f.write(f"{test_id}|{user_query}|{results}\n")
