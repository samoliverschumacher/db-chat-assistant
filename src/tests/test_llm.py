################################################################################################
#           TEST IF RESPONSE CONTAINS THE CONTEXT IT WAS GIVEN
################################################################################################
# Checks if the synthesized response can be deduced from the sources. If not, there is halluciniation.

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, and an instruction
# for the llm to decide if the response could be given using the context information alone.

# Call the llm with the prompt.

# Score the model - "1" if it said there was enough info, "0.5" if partially, and "0" if not.


################################################################################################
#           TEST CONTEXT RETRIEVAL
################################################################################################
# Given a query, use a LLM to decide if the context retrieved contains enough to answer the query

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, and an instruction 
# for the llm to decide if the context information has enough in it to answer the users query.

# Call the llm with the prompt.

# Post-process the response, score the result as "1" if it said there was enough info, "0.5" if partially, 
# and "0" if not.


################################################################################################
#           TEST IF RESPONSE & SOURCE CONTEXT ANSWERS THE QUERY
################################################################################################
# 

# Load an example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context (document chunks), the original query, the synthesized response
# and an instruction for the llm to decide if the response answers the query.

# Call the llm with the prompt.

# Score the model - "1" if it said there was enough info, "0.5" if partially, and "0" if not.

# Compose the prompt with the context (document chunks), the original query, the synthesized response
# and an instruction for the llm to decide if each of the context sources contains an answer to the query


################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - CLASSICAL NLP TECHNIQUES
################################################################################################

# Use ROGUE metric to compare synthesized answers against human crafted answers
# https://huggingface.co/spaces/evaluate-metric/rouge


################################################################################################
#           TEST SYNTHESIZED RESPONSE AGAINST HUMAN CRAFTED ANSWER - LLM AS JUDGE
################################################################################################

# Load example user query

# Initialise the encoder (llm)

# Retrieve
#     Encode the query

#     load the array of document vectors

#     flatten the array of chunks of documents, to a 1D array of document-chunks

#     Compare the encoded query to all the document-chunks

#     identify the most similar document-chunk

#     identify the chunk before, and chunk after the most similar document-chunk, if they exist.

#     Load from the vector database the most similar document-chunk, and the before and after chunks.

# Compose the prompt with the context

# Get the LLMs response

# Compose a prompt with the original query and the synthesized response for the "Judge" LLM

# Get a score for the user query from the "Judge" LLM

# Save the score and the explanation of the score to file.


################################################################################################
#           TEST SYNTHESIZED RESPONSE - MANUAL EVALUATION
################################################################################################
# For each query-synthesized response pair, give a "LIKE" or "DISLIKE"
