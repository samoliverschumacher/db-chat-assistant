import os
from datetime import datetime
from pathlib import Path
import subprocess
import logging

import pandas as pd
from dbchat.langchain_agent import LangchainAgent

from dbchat.logger import GitLogger as Logger
        
        
if __name__ == '__main__':
    
    log_dir = Path("Experiments", "logs", "pandas_agent")
    logger = Logger(log_dir)
    
    os.environ["OPENAI_API_KEY"] = "api-key-goes-here"

    testdatapath = Path(__file__).parent / "../src/tests/data/end-to-end.csv"
    with open(testdatapath) as f:
        test_data = f.readlines()
    
    for id, user_query, tables, _ in test_data:

        dfs=[]
        for table in tables.split(","):
            db_path = str(Path("../data/chinook.db").resolve())
            dfs.append(pd.read_sql(f"SELECT * FROM {table}",
                        con=db_path))
        
        agent = LangchainAgent(dfs, logger=logger)

        # Call the ask method to get the results
        results = agent.ask(user_query)
        print(results)

