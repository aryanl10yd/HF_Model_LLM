from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import login, HfApi, HfFolder
import requests
import torch

# login(token="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")

api_token = "YOUR_API_TOKEN_HERE"

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_model(user_input, history=[]):
    # Encode user input and append to history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids if history == [] else torch.cat([history, new_user_input_ids], dim=-1)

    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated response and update history
    chat_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    history = chat_history_ids

    return chat_output, history

# Example chat
history = []
user_input = "Hello, how are you?"
response, history = chat_with_model(user_input, history)
print("Bot:", response)


