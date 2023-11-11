

def load_example_queries(test_data_path):   
    """
    Expects a csv like;
    id|user_query|tables|note
    1|How much money have we made in Berlin?|invoices|chooses the correct table.
    """
    test_data = []
    with open(test_data_path) as f:
        f.readline()  # Remove header row
        for row in f.readlines():
            id, user_query, tables, comment = row.split('|')
            test_data.append((id, user_query, tables, comment))
