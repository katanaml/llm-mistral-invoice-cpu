import sys
import timeit

from dotenv import load_dotenv
load_dotenv()
import os

import box
import yaml

from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from langchain.llms.llamacpp import LlamaCpp

from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

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

model_local_path= "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
DB_URL=os.getenv('POSTGRES_DB_URL')

print(f'\nModel used: {model_local_path}')
print(f'\nDB URL: {DB_URL}')

start = timeit.default_timer()

set_llm_cache(SQLiteCache(database_path="langchain_cache.db"))

llm = LlamaCpp(
    model_path=model_local_path,
    temperature=0.2,
    max_tokens=8000,
    top_p=1,
    verbose=False,
    n_ctx=4096
)

db = SQLDatabase.from_uri(
    f"{DB_URL}",
    include_tables=['projects']
)

schema = db.get_table_info(table_names=['projects'])

chain = SQLDatabaseChain.from_llm(llm=llm,db=db,verbose=False)

QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer. If you can not find any answer do not start assuming stuff, return 'None' instead.
Use the following format:
Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Use the following info for your ease in generating queries:
{schema}

{question}
"""

message = QUERY.format(question=message,schema=schema)
result = chain.run(message)

end = timeit.default_timer()

print(f'\nAnswer: {result}')
print('='*180)

print(f"Time to retrieve answer: {end - start}")