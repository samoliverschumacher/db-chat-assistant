# pip requirements: 
# fuzzywuzzy==0.18.0
# spacy==3.6.1

import spacy
from spacy.tokens import Token
from fuzzywuzzy import fuzz

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Register the 'coref' extension
Token.set_extension("coref", default=None)

# Example tables in the database
tables = {
    "employees": ["id", "name", "department"],
    "projects": ["id", "name", "start_date", "end_date"],
    "departments": ["id", "name", "location"],
}

# Example prompts
prompts = [
    "Show me all employees in the Sales department.",
    "Find projects that started after 2022.",
    "Get the names of all departments.",
]

# Process the prompt and match with database entities
for prompt in prompts:
    doc = nlp(prompt)

    # Named Entity Recognition (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Dependency Parsing
    dependencies = [(token.text, token.dep_) for token in doc]

    # Relation Extraction
    relations = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]

    # Coreference Resolution
    coreferences = [(token.text, token._.coref) for token in doc]

    # Match table names
    table_names = []
    for token in doc:
        for table in tables.keys():
            similarity = fuzz.ratio(token.text.lower(), table.lower())
            if similarity >= 80:  # Adjust the similarity threshold as needed
                table_names.append(table)

    # Match field names
    field_names = []
    for token in doc:
        for fields in tables.values():
            for field in fields:
                similarity = fuzz.ratio(token.text.lower(), field.lower())
                if similarity >= 80:  # Adjust the similarity threshold as needed
                    field_names.append(field)

    print("Prompt:", prompt)
    print("Entities:", entities)
    print("Dependencies:", dependencies)
    print("Relations:", relations)
    print("Coreferences:", coreferences)
    print("Table Names:", table_names)
    print("Field Names:", field_names)
    print()
