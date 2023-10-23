import sys
import timeit
from ctransformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent,AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain
import os
from dotenv import load_dotenv


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
# model_local_path = "./models/mistral-7b-instruct-v0.1.Q2_K.gguf"
model_local_path= "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

# model_name = model_mistral_Q2_K
# model_local_path= "./models/{model_name}"

# print(f'\nModel used: {model_local_path}')

start = timeit.default_timer()
# llm = AutoModelForCausalLM.from_pretrained(
#         # model_TheBloke,
#         # model_file=model_mistral_Q5_K_M,
#         model_type=model_type_mistral,
#         gpu_layers=0,
#         # local_files_only=True,
#         model_path_or_repo_id=model_local_path
# )
# prompt = PromptTemplate.from_template(
#     """You are an expert AI assistant that helps user's with friendly and detailed answers

# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """
# )
# message= prompt.format(question=message)
# result = llm(message)
load_dotenv()
open_ai_key= os.getenv("OPENAI_API_KEY")

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://s8user%40s8postgres:3%5EcZ%218uWZ%2563mU@s8postgres.postgres.database.azure.com:5432/ncrportaldb_v2",
    # f"postgresql+psycopg2://{(username)}:{(password)}@{(host)}:5432/{(database)}",
)

llm = ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0, openai_api_key=open_ai_key)

# memory = SQLChatMessageHistory(
#     session_id='test_session',
#     connection_string='sqlite:///sqlite.db'
# )
# entity_store = SQLiteEntityStore(db_file='sqlite.db')
# agent = create_sql_agent(
#     llm=llm,
#     toolkit=SQLDatabaseToolkit(db=db,llm=llm),
#     # verbose=True,
#     agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     handle_parsing_errors=True
#     # memory=memory
#     )
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
result = chain.run(message)
print(result)


end = timeit.default_timer()

print(f'\nAnswer: {result}')
print('='*150)

print(f"Time to retrieve answer: {end - start}")