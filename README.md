# LORA_Navi3
finetuning and inferencing code for LORA on Navi3


# How To Run

## Environment set up
```
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --name=lora_navi3 --shm-size=8g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --group-add video rocm/pytorch:latest

apt install hipblaslt
pip install --upgrade pip
pip install einops lion_pytorch accelerate
pip install git+https://github.com/ROCm/transformers.git
git clone --recurse https://github.com/ROCmSoftwarePlatform/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled
make hip
python setup.py install

pip install datasets
pip install peft
pip install trl
```

## Finetuning
```
git clone https://github.com/Lzy17/LORA_Navi3.git
cd LORA_Navi3
python finetune.py
```

## Inferencing
```
export TOKENIZERS_PARALLELISM=false
python inference.py
```



# Reference Output
```
=================NousResearch/Llama-2-7b-chat-hf outputs=================
<s>[INST] What do you think is the most important part of building an AI chatbot? [/INST]  There are several important aspects to consider when building an AI chatbot, but here are some of the most critical elements:

1. Natural Language Processing (NLP): A chatbot's ability to understand and interpret human language is crucial for effective communication. NLP is the foundation of any chatbot, and it involves training the AI model to recognize patterns in language, interpret meaning, and generate responses.
2. Conversational Flow: A chatbot's conversational flow refers to the way it interacts with users. A well-designed conversational flow should be intuitive, easy to follow, and adaptable to different user scenarios. This involves creating a dialogue flow that is logical, coherent, and engaging.
3. Domain Knowledge: A chatbot's domain knowledge refers to the



=================NousResearch/Llama-2-7b-chat-hf with LORA finetuned outputs=================
<s>[INST] What do you think is the most important part of building an AI chatbot? [/INST] There are several important parts to building an AI chatbot, but some of the most critical include:

1. Natural Language Processing (NLP): The ability of the chatbot to understand and interpret human language is crucial for effective communication. NLP is a key component of chatbots, and it allows them to understand the context and intent behind a user's message.
2. Machine Learning (ML): ML is the technology that enables chatbots to learn and improve over time. By analyzing user interactions and feedback, ML algorithms can help chatbots adapt and improve their responses to better meet user needs.
3. Knowledge Base: A chatbot's knowledge base is the collection of information and data that it uses to answer user questions and provide responses. A well-designed knowledge base is essential for ensuring that
```
