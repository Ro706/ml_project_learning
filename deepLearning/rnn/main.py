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
text = response.text[:20000]  # Use first 20k chars to keep training fast
print(f"Dataset loaded! Total characters: {len(text)}")
print(f"Sample text: {text[:100]}\n")

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
print(f"Vocabulary size: {vocab_size} unique characters")

# 2. Hyperparameters
seq_length = 5
embedding_dim = 16
hidden_size = 128
learning_rate = 0.01
epochs = 150

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

# 4. RNN Architecture
class AdvancedCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(AdvancedCharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])  # Take last timestep output
        return out

model = AdvancedCharRNN(vocab_size, embedding_dim, hidden_size)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created! Total parameters: {total_params:,}")

# 5. Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\nTraining on {len(X)} sequences for {epochs} epochs...")
print("-" * 50)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("-" * 50)
print("Training complete!")

# 6. Generation with Temperature Sampling
def sample_with_temperature(logits, temperature=1.0):
    logits = logits.squeeze(0)               # Shape: [vocab_size]
    if temperature <= 0:
        return torch.argmax(logits).item()
    logits = logits / temperature
    probs = F.softmax(logits, dim=0)
    return torch.multinomial(probs, num_samples=1).item()

model.eval()

# Pick a valid start sequence from the actual text
start_seq = text[:seq_length]  # Guaranteed to be in vocabulary
generated_text = start_seq

print(f"\nStarting sequence: '{start_seq}'")
print("\nGenerating Text (Temperature 0.8):")
print("-" * 50)

with torch.no_grad():
    for _ in range(200):  # Generate 200 characters
        window = generated_text[-seq_length:]

        try:
            x_input = [char_to_index[char] for char in window]
        except KeyError as e:
            print(f"\nError: Character {e} not found in training vocabulary.")
            break

        x_input_tensor = torch.tensor([x_input], dtype=torch.long)

        prediction = model(x_input_tensor)        # Shape: [1, vocab_size]
        next_index = sample_with_temperature(prediction, temperature=0.8)
        next_char = index_to_char[next_index]

        generated_text += next_char

print(generated_text)
print("-" * 50)

# 7. Try different temperatures
print("\nSame start, different temperatures:")
for temp in [0.2, 0.5, 1.0, 1.5]:
    generated = start_seq
    with torch.no_grad():
        for _ in range(100):
            window = generated[-seq_length:]
            x_input = [char_to_index[char] for char in window]
            x_input_tensor = torch.tensor([x_input], dtype=torch.long)
            prediction = model(x_input_tensor)
            next_index = sample_with_temperature(prediction, temperature=temp)
            generated += index_to_char[next_index]
    print(f"\nTemp {temp}: {generated}")