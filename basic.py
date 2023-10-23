import sys
import timeit
from ctransformers import AutoModelForCausalLM

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
result = llm(message)
end = timeit.default_timer()

print(f'\nAnswer: {result}')
print('='*150)

print(f"Time to retrieve answer: {end - start}")