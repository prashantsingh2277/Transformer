import torch
from torch import nn, optim
from dataset import get_dataloader
from model import Transformer
from utils import save_model
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

file_path = "brown.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_vocab_size = trg_vocab_size = 30522
embed_size = 1024
num_layers = 8
heads = 16
forward_expansion = 4
dropout = 0.1
src_pad_idx = 0
max_length = 200
learning_rate = 3e-4
batch_size = 4
num_epochs = 10
accumulation_steps = 2  

# Load data
train_loader = get_dataloader(file_path, batch_size, max_length)

# Initialize model
model = Transformer(
    src_vocab_size, trg_vocab_size, src_pad_idx, embed_size, num_layers,
    forward_expansion, heads, dropout, device, max_length
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# Training loop
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for i, batch in enumerate(train_loader):
        # Loop through the batch and access each element correctly
        src = batch["input_ids"].to(device)  # Assuming batch is a dictionary with 'input_ids'
        trg = batch["input_ids"].to(device)  # Assuming the same for target sequence
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(src, trg[:, :-1])  # Exclude last token for teacher forcing
            output = output.float()  # Ensuring the output is in float32 before masking
            loss = criterion(output.reshape(-1, output.shape[2]), trg[:, 1:].reshape(-1))  # Shift target by 1

        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            avg_loss = running_loss / accumulation_steps
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {avg_loss:.4f}")
            running_loss = 0  # Reset after logging

    scheduler.step()  # Step after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Learning rate: {scheduler.get_last_lr()[0]:.6f}")

# Save model
save_model(model, optimizer, "transformer_model.pth")
