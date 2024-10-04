import torch
import tiktoken
from dataloader import create_dataloader
from gpt import GPTModel
from utils import evaluate_model

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 128, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 24,         # Number of attention heads
    "n_layers": 24,        # Number of layers
    "drop_rate": 0.0,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

tokenizer = tiktoken.get_encoding("gpt2")

with open("dataset.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

train_dataloader, val_dataloader = create_dataloader(
    txt=text_data,
    ratio=0.8,
    batch_size=8,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)


model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 200

for epoch in range(num_epochs):
    for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output.flatten(0, 1), target_batch.flatten())
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            train_loss, val_loss = evaluate_model(
                model, train_dataloader, val_dataloader, device, 5)
            print(f"Ep {epoch+1}: "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
print('Finish!')

torch.save(model.state_dict(), "model.pt")

