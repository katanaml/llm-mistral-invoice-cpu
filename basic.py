import sys
import timeit
import yaml

import box
import yaml

from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain

from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Get query from console
if len(sys.argv) > 1:
    a = 1
else:
    print("missing required argument 'message'")
    sys.exit

message = sys.argv[1]


model_TheBloke = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model_mistral_Q5_K_M = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
model_mistral_Q2_K = "mistral-7b-instruct-v0.1.Q2_K.gguf"
model_type_mistral = "mistral"
model_type_llama = "llama"
model_local_path_Q2_K = "./models/mistral-7b-instruct-v0.1.Q2_K.gguf"
model_local_path= "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

print(f'\nModel used: {model_local_path}')

start = timeit.default_timer()

llm = LlamaCpp(
    model_path=model_local_path,
    temperature=0.75,
    max_tokens=8000,
    top_p=1,
    # callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# print(llm("AI is going to"))

# prompt = """
# Question: A rap battle between Stephen Colbert and John Oliver
# """
# llm(prompt)

db = SQLDatabase.from_uri(
    f"{cfg.POSTGRES_DB_URL}",
    include_tables=["_prisma_migrations"]
    # f"postgresql+psycopg2://{(username)}:{(password)}@{(host)}:5432/{(database)}",
)

chain = SQLDatabaseChain.from_llm(llm=llm,db=db,verbose=True)

QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer. If you can not find any answer do not start assuming stuff, return 'None' instead.
Use the following format:
Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""
message = QUERY.format(question=message)
result = chain.invoke(message)

end = timeit.default_timer()


print(f'\nAnswer: {result}')
print('='*180)

print(f"Time to retrieve answer: {end - start}")