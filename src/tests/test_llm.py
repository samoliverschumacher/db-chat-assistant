################################################################################################
#           TEST CONTEXT RETRIEVAL
################################################################################################

# Load an example user query

# Initialise the encoder (llm)

# Encode the query

# load the array of document vectors

# flatten the array of chunks of documents, to a 1D array of document-chunks

# Compare the encoded query to all the document-chunks

# identify the most similar document-chunk

# identify the chunk before, and chunk after the most similar document-chunk, if they exist.

# Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, and an instruction 
# for the llm to decide if the context information has enough in it to answer the users query.

# Call the llm with the prompt.

# Post-process the response, score the result as "1" if it said there was enough info, "0.5" if partially, and "0" if not.


################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - CLASSICAL NLP TECHNIQUES
################################################################################################

# Use ROGUE metric to compare synthesized answers against human crafted answers
# https://huggingface.co/spaces/evaluate-metric/rouge


################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - LLM AS JUDGE
################################################################################################

# Load example user query

# Retrieve the context from the vector database

# Compose the prompt with the context

# Get the LLMs response

# Compose a prompt with the original query and the synthesized response for the "Judge" LLM

# Get a score for the user query from the "Judge" LLM

# Save the score and the explanation of the score to file.
