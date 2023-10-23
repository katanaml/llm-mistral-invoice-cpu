import sys
import timeit
from ctransformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate

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

print(f'\nModel used: {model_local_path}')

start = timeit.default_timer()
llm = AutoModelForCausalLM.from_pretrained(
        # model_TheBloke,
        # model_file=model_mistral_Q5_K_M,
        model_type=model_type_mistral,
        gpu_layers=0,
        # local_files_only=True,
        model_path_or_repo_id=model_local_path
)
prompt = PromptTemplate.from_template(
    """You are an expert AI assistant that helps user's with friendly and detailed answers

Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
)
message= prompt.format(question=message)
result = llm(message)
end = timeit.default_timer()

print(f'\nAnswer: {result}')
print('='*150)

print(f"Time to retrieve answer: {end - start}")