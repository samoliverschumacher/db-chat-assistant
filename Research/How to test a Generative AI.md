In testing the system is giving the correct results, What should our tests be focusing on?

What do we assume already works for an LLM?
- i.e. "Checking the agent can recall info from a length N context", has already been done - we dont want to test the dependencies of our project.

We test the user intent, but only pass the user query into the system.
- User intents such as "Intent 1: Describe the answer, Intent 2: Reason about the answer", might map to a larger number of possible inputs. And those inputs might map to a larger number of expected outputs