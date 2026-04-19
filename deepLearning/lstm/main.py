import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import requests

# 1. Data Preparation - Tiny Shakespeare Dataset
print("Downloading Tiny Shakespeare dataset...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text[:20000]
print(f"Dataset loaded! Total characters: {len(text)}")
print(f"Sample text: {text[:100]}\n")

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
print(f"Vocabulary size: {vocab_size} unique characters")

# 2. Improved Hyperparameters
seq_length = 30
embedding_dim = 32
hidden_size = 256
num_layers = 2
dropout = 0.3
learning_rate = 0.002
epochs = 300
batch_size = 128

# 3. Sequencing
print("\nPreparing sequences...")
sequences = []
labels = []
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)
print(f"Total sequences created: {len(X)}")

# 4. Upgraded LSTM Architecture
class AdvancedCharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, drop):
        super(AdvancedCharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=drop if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        out, hidden = self.lstm(embedded, hidden)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = AdvancedCharLSTM(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created! Total parameters: {total_params:,}")

# 5. Training Loop with Mini-Batching + LR Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# FIX: Removed 'verbose=True' — not supported in PyTorch 2.x
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"\nTraining on {len(X)} sequences for {epochs} epochs...")
print(f"Batch size: {batch_size} | Batches per epoch: {len(dataloader)}")
print("-" * 50)

best_loss = float('inf')
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs, _ = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)

    # FIX: Manually print LR instead of relying on verbose=True
    if (epoch + 1) % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.5f}')

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')

print("-" * 50)
print(f"Training complete! Best loss: {best_loss:.4f}")

# Load best model for generation
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# 6. Generation with Temperature Sampling
def sample_with_temperature(logits, temperature=1.0):
    logits = logits.squeeze(0)
    if temperature <= 0:
        return torch.argmax(logits).item()
    logits = logits / temperature
    probs = F.softmax(logits, dim=0)
    return torch.multinomial(probs, num_samples=1).item()

def generate_text(model, start_seq, length=300, temperature=0.8):
    model.eval()
    generated = start_seq

    # Pad if start_seq is shorter than seq_length
    if len(start_seq) < seq_length:
        start_seq = start_seq.rjust(seq_length)

    with torch.no_grad():
        for _ in range(length):
            window = generated[-seq_length:]
            try:
                x_input = [char_to_index[c] for c in window]
            except KeyError as e:
                print(f"Error: Character {e} not in vocabulary.")
                break

            x_tensor = torch.tensor([x_input], dtype=torch.long).to(device)
            prediction, _ = model(x_tensor)
            next_index = sample_with_temperature(prediction, temperature)
            generated += index_to_char[next_index]

    return generated

# 7. Results
start_seq = text[:seq_length]
print(f"\nStarting sequence: '{start_seq}'")

print("\n" + "="*50)
print("Generating Text (Temperature 0.8):")
print("="*50)
print(generate_text(model, start_seq, length=300, temperature=0.8))

print("\n" + "="*50)
print("Temperature Comparison (100 chars each):")
print("="*50)
for temp in [0.2, 0.5, 1.0, 1.5]:
    result = generate_text(model, start_seq, length=100, temperature=temp)
    print(f"\nTemp {temp}:\n{result}")
    print("-" * 40)