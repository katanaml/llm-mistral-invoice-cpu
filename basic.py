import sys
import timeit

from langchain.llms import ctransformers, LlamaCpp
from langchain.llms import huggingface_pipeline

from ctransformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent,AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.sql import SQLDatabaseChain

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from transformers import MistralForCausalLM


from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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


# model = AutoModelForCausalLM.from_pretrained(
#     model_TheBloke,
#     # pretrained_model_name_or_path=model_local_path,
#     model_type=model_type_mistral,
#     # model_file=model_mistral_Q5_K_M,
#     local_files_only=True,
#     gpu_layers=0
# )

# # print(model("AI is going to"))

# tokenizer = AutoTokenizer.from_pretrained(
#     model_TheBloke,
#     # pretrained_model_name_or_path=model_local_path,
#     # model_type=model_type_llama,
#     # model_file=model_mistral_Q5_K_M,
#     # local_files_only=True,
#     # gpu_layers=0
# )

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="< path to the GGUF file you downloaded >",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    # callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm(prompt)

start = timeit.default_timer()

# llm = huggingface_pipeline(
#         model_TheBloke,
#         model=model,
#         tokenizer=tokenizer,
#         # model_file=model_mistral_Q5_K_M,
#         model_type=model_type_mistral,
#         gpu_layers=0,
#         local_files_only=True,
#         model_path_or_repo_id=model_local_path,
#         task="text-generation",
#         # pipeline_kwargs={"max_new_tokens": 10},
#         # model_kwargs={"temperature": 0, "max_length": 64, 'device': 'cpu'}
# )

# llm = vars(llm)
#  hf = HuggingFacePipeline.from_model_id(
#                 model_id="gpt2",
#                 task="text-generation",
#                 pipeline_kwargs={"max_new_tokens": 10},
#             )

# hf = huggingface_pipeline(
#     pipeline=pipe,
#     # local_files_only=True
#     gpu_layers=0,
# )

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://s8user%40s8postgres:3%5EcZ%218uWZ%2563mU@s8postgres.postgres.database.azure.com:5432/ncrportaldb_v2",
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
result = chain.run(message)

end = timeit.default_timer()

print(f'\nAnswer: {result}')
print('='*180)

print(f"Time to retrieve answer: {end - start}")