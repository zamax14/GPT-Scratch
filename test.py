import torch
import tiktoken
from gpt import GPTModel
from utils import (
    token_ids_to_text,
    text_to_token_ids,
    generate_text_simple
)

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 128, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 24,         # Number of attention heads
    "n_layers": 24,        # Number of layers
    "drop_rate": 0.0,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

tokenizer = tiktoken.get_encoding('gpt2')
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model_good.pt"))

test_txt = 'Que es la masa madre? \n'
txt_tokens = text_to_token_ids(text=test_txt, tokenizer=tokenizer)
output_tokens = generate_text_simple(model=model, idx=txt_tokens, max_new_tokens=100, context_size=GPT_CONFIG_124M["context_length"])
output_text = token_ids_to_text(token_ids=output_tokens, tokenizer=tokenizer)
print(output_tokens)
print(output_text)