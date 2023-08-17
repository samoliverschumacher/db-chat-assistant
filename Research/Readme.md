
## Candidate Solutions
The investigation of candidate solutions are available in [LINK](https://docs.google.com/document/d/1rMF0u5UpWW0RSRuBa4COEP27F_KjACgCC6jtW9afEK0/edit)
After finalizing, it will be reorganized in this document.
### Challenges  
- How does the agent decide which table to look at?  
- Calibration of the system to the user  
- How complicated can an agents insights of a database be, if all it has as context is table metadata? i.e. how close to a full blown EDA can it get?  
- Calibration of the system to the user  
- Large amounts of tabular data isn't what llm's are optimised for.  
- How to create embedding vectors for various parts of database? i.e. do we create embeddings based on utility for a llm, per table, or per [semantic layer]([https://www.dremio.com/resources/guides/what-is-a-semantic-layer/#:~:text=The%20semantic%20layer%20is%20a,tools%20for%20your%20end%20users.](https://www.dremio.com/resources/guides/what-is-a-semantic-layer/#:~:text=The%20semantic%20layer%20is%20a,tools%20for%20your%20end%20users.))?  
- How do we evaluate the success of the system? i.e. test data?  
- Integrations to all the DB's  
- Dirty data can lead to poor quality responses. (see "calibration")