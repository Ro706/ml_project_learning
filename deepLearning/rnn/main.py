import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 1. Data Preparation
# Note: Case matters! 'H' is different from 'h'.
text = "Hello everyone my name is Rohit."
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# 2. Advanced Hyperparameters
seq_length = 5
embedding_dim = 16
hidden_size = 128
learning_rate = 0.01
epochs = 150

# 3. Sequencing
sequences = []
labels = []
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

# 4. Advanced RNN Architecture
class AdvancedCharRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(AdvancedCharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        return out

model = AdvancedCharRNN(vocab_size, embedding_dim, hidden_size)

# 5. Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training on {len(X)} sequences...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. Advanced Generation with Temperature
def sample_with_temperature(logits, temperature=1.0):
    if temperature <= 0:
        return torch.argmax(logits, dim=1).item()
    logits = logits / temperature
    probs = F.softmax(logits, dim=1)
    return torch.multinomial(probs, num_samples=1).item()

model.eval()

# --- FIX: start_seq must only use characters present in 'text' ---
start_seq = "Hello" 
generated_text = start_seq

print("\nGenerating Text (Temperature 0.8):")
with torch.no_grad():
    for _ in range(50):
        # Slice the last 'seq_length' characters
        window = generated_text[-seq_length:]
        
        # Convert window to indices
        try:
            x_input = [char_to_index[char] for char in window]
        except KeyError as e:
            print(f"\nError: Character {e} not found in training vocabulary.")
            break
            
        x_input_tensor = torch.tensor([x_input], dtype=torch.long)

        # Predict and sample
        prediction = model(x_input_tensor)
        next_index = sample_with_temperature(prediction, temperature=0.8)
        next_char = index_to_char[next_index]

        generated_text += next_char

print(generated_text)
