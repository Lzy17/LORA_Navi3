import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
from peft import get_peft_model

from peft import LoraConfig, PeftModel

import warnings
warnings.filterwarnings("ignore")

# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_name = "./llama-2-7b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
#model = PeftModel.from_pretrained(base_model, new_model_name)
#model = model.merge_and_unload()


# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Generate text using base model
print("=================NousResearch/Llama-2-7b-chat-hf outputs=================")
query = "What do you think is the most important part of building an AI chatbot?"
text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print(output[0]['generated_text'])

print('\n'*2)

# Generate text using fine-tuned model
print("=================NousResearch/Llama-2-7b-chat-hf with LORA finetuned outputs=================")
base_model.load_adapter(new_model_name)
query = "What do you think is the most important part of building an AI chatbot?"
text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_length=200)
output = text_gen(f"<s>[INST] {query} [/INST]")
print(output[0]['generated_text'])
