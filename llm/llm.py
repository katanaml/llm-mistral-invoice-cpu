import box
import yaml

from langchain.llms import ctransformers
from langchain.llms import huggingface_pipeline
from langchain.llms import huggingface_hub

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM

from ctransformers import AutoModelForCausalLM

from langchain.chains import LLMChain

from llm.prompts import qa_template


from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

model_TheBloke = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model_mistral_Q5_K_M = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
model_local_dir = "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
model_type_mistral="mistral"
model_local_path= "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"


model_id_TheBloke_Q5_K_M = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
model_id_google = 'google/flan-t5-small'
model_id_mistral = "mistralai/Mistral-7B-Instruct-v0.1"

def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

def setup_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        model_TheBloke,
        # model_file=model_mistral_Q5_K_M,
        model_type=model_type_mistral,
        gpu_layers=0,
        local_files_only=True
    )
    return llm
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    # hf = huggingface_pipeline(pipeline=pipe)

    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # pipe = pipeline(
    #     "text-generation",
    #     model=model, 
    #     tokenizer=tokenizer, 
    #     token=128
    # )

    # local_llm = huggingface_pipeline(pipeline=pipe)

    # print(local_llm)
    # return local_llm

    # lm = huggingface_pipeline.from_model_id(
    #     model_id,
    #     task="text-generation",
    #     model_kwargs={"temperature": 0, "max_length": 64},
    # )
    

    # print(lm)
    # return lm

    # llm = huggingface_pipeline.HuggingFacePipeline.from_model_id(
    #     model_id=model_id_TheBloke,
    #     task="text-generation",
    #     model_kwargs={"temperature": 0.7, "do_sample":True},
    # )
    # llm = huggingface_hub.HuggingFaceHub(repo_id=model_id_TheBloke,model_kwargs={"temperature":0.7},task="text-generation",verbose=True)
    # result = llm.predict("tell me a joke")
    # print(result)
    # return llm
    # chain = LLMChain(llm=llm)
    # chain.run("tell me a joke")

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    # lm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)
   
    # qa_prompt = set_qa_prompt()

    # print(llm("tell me a joke"))
    # result =llm(qa_prompt)
    # print(result)
    # return result
    
