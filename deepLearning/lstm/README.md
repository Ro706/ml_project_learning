# 🎭 Character-Level LSTM Text Generation — Tiny Shakespeare

> A deep dive into Long Short-Term Memory networks, trained to generate Shakespeare-like text character by character.

---

## 📚 Table of Contents

1. [What is an LSTM?](#what-is-an-lstm)
2. [LSTM vs RNN](#lstm-vs-rnn)
3. [LSTM Gates — How Memory Works](#lstm-gates)
4. [Project Architecture](#project-architecture)
5. [The Code](#the-code)
6. [Training Diagrams & Results](#training-results)
7. [Generated Text](#generated-text)
8. [How to Run](#how-to-run)

---

## 🧠 What is an LSTM?

**LSTM (Long Short-Term Memory)** is a special type of Recurrent Neural Network (RNN) designed to learn and remember information over **long sequences**. It was introduced by Hochreiter & Schmidhuber in 1997 to solve the problem of RNNs forgetting information from many steps back.

A standard neural network processes one input and produces one output — it has no memory. An LSTM processes sequences step by step, maintaining a **hidden state** (short-term memory) and a **cell state** (long-term memory) that carry information across all timesteps.

```
                       ┌─────────────────────────────────────────┐
                       │           LSTM CELL                     │
                       │                                         │
  Cell State ─────────►│──────────────────────────────────────►──│──► Cell State
  (Long-term memory)   │     ×           +           ×           │    (updated)
                       │     │           │           │           │
                       │  Forget      Input        Output        │
                       │   Gate        Gate         Gate         │
                       │     │           │           │           │
  Hidden State ───────►│─────┴───────────┴───────────┴─────────►─│──► Hidden State
  (Short-term memory)  │                                         │    (updated)
                       └─────────────────────────────────────────┘
                                         ▲
                                      Input Xₜ
```

---

## ⚔️ LSTM vs RNN

| Feature | Vanilla RNN | LSTM |
|---|---|---|
| Memory type | Only hidden state | Hidden state + Cell state |
| Long-range memory | ❌ Forgets quickly | ✅ Remembers long sequences |
| Vanishing gradient | ❌ Severe problem | ✅ Largely solved |
| Gate mechanism | None | Forget, Input, Output gates |
| Parameters | Few | More (~4x RNN) |
| Training stability | Unstable | Much more stable |
| Best for | Very short sequences | Long sequences, text, speech |

### Why RNNs Fail on Long Sequences

When an RNN processes a long sequence, gradients must flow backwards through every timestep. They get **multiplied by small numbers repeatedly**, shrinking to near zero — the model stops learning from early inputs.

```
RNN Gradient Flow (vanishing):

Timestep:  t=1    t=2    t=3    t=4    t=5  ... t=30
Gradient:  0.001  0.01   0.1    0.5    1.0  ← only recent steps matter

LSTM Gradient Flow (stable):

Timestep:  t=1    t=2    t=3    t=4    t=5  ... t=30
Gradient:  0.9    0.9    0.9    0.9    1.0  ← cell state preserves gradients
```

---

## 🚪 LSTM Gates — How Memory Works

An LSTM cell has **3 gates** that control what information is stored, discarded, or passed on. Each gate uses a **sigmoid function** (outputs 0 to 1) — think of it as a valve: 0 = fully closed, 1 = fully open.

---

### Gate 1 — Forget Gate 🗑️

Decides **what to throw away** from the cell state (long-term memory).

```
         Hidden State (hₜ₋₁)
                │
Input (Xₜ) ────┤
                │
                ▼
           [Sigmoid σ]   ← outputs 0.0 to 1.0
                │
                ▼
         Forget Value fₜ
                │
                ▼
    Cell State × fₜ  ──► 0 = forget everything
                          1 = keep everything
```

**Formula:** `fₜ = σ(Wf · [hₜ₋₁, Xₜ] + bf)`

**Example:** When reading "The cat sat on the... **dog**", the forget gate discards "cat" from memory so "dog" becomes the new subject.

---

### Gate 2 — Input Gate ✍️

Decides **what new information to add** to the cell state.

```
         Hidden State (hₜ₋₁)
                │
Input (Xₜ) ────┤
               / \
              /   \
             ▼     ▼
        [Sigmoid] [Tanh]
             │       │
         iₜ (how  C̃ₜ (what
          much)   to add)
             │       │
             └───×───┘
                 │
                 ▼
          New information
          added to cell state
```

**Formulas:**
- `iₜ = σ(Wi · [hₜ₋₁, Xₜ] + bi)`  ← how much to write
- `C̃ₜ = tanh(Wc · [hₜ₋₁, Xₜ] + bc)` ← what to write

---

### Gate 3 — Output Gate 📤

Decides **what to output** as the new hidden state.

```
  Updated Cell State Cₜ
           │
           ▼
         [Tanh]   ← squashes to -1 to 1
           │
           │    Hidden State (hₜ₋₁)
           │           │
           │    Input (Xₜ) ──┤
           │                 ▼
           │           [Sigmoid σ]
           │                 │
           └────────×────────┘
                    │
                    ▼
            New Hidden State hₜ
            (also the output)
```

**Formulas:**
- `oₜ = σ(Wo · [hₜ₋₁, Xₜ] + bo)`
- `hₜ = oₜ × tanh(Cₜ)`

---

### Full LSTM Cell Update Summary

```
┌─────────────────────────────────────────────────────────┐
│  Given: previous hidden state hₜ₋₁, cell state Cₜ₋₁    │
│         and current input Xₜ                            │
│                                                         │
│  Step 1 — Forget:  fₜ  = σ(Wf·[hₜ₋₁,Xₜ] + bf)        │
│  Step 2 — Input:   iₜ  = σ(Wi·[hₜ₋₁,Xₜ] + bi)        │
│                    C̃ₜ  = tanh(Wc·[hₜ₋₁,Xₜ] + bc)     │
│  Step 3 — Update:  Cₜ  = fₜ×Cₜ₋₁ + iₜ×C̃ₜ            │
│  Step 4 — Output:  oₜ  = σ(Wo·[hₜ₋₁,Xₜ] + bo)        │
│                    hₜ  = oₜ × tanh(Cₜ)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Project Architecture

### Data Flow

```
Raw Text:  "First Citizen: Before we proceed..."
               │
               ▼
    ┌─────────────────────┐
    │   Character Vocab   │  58 unique chars
    │  'A'→0, 'B'→1 ...  │
    └─────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Sliding Window     │  seq_length = 30
    │  "First Citizen: Be"│ → label: "f"
    │  "irst Citizen: Bef"│ → label: "o"
    └─────────────────────┘
               │
               ▼  19,970 sequences
    ┌─────────────────────────────────────────┐
    │             MODEL                       │
    │                                         │
    │  [30 char indices]                      │
    │        ↓                                │
    │  Embedding Layer  (58 → 32)             │
    │        ↓                                │
    │  Dropout (0.3)                          │
    │        ↓                                │
    │  LSTM Layer 1     (32 → 256)            │
    │        ↓                                │
    │  LSTM Layer 2     (256 → 256)           │
    │        ↓                                │
    │  Last Timestep    out[:, -1, :]         │
    │        ↓                                │
    │  Dropout (0.3)                          │
    │        ↓                                │
    │  Linear Layer     (256 → 58)            │
    │        ↓                                │
    │  Logits → Softmax → Next Character      │
    └─────────────────────────────────────────┘
```

### Model Parameters Breakdown

```
Layer                    Shape              Parameters
─────────────────────────────────────────────────────
Embedding                58 × 32              1,856
LSTM Layer 1             32→256 (×4 gates)  ~263,168
LSTM Layer 2             256→256 (×4 gates) ~525,312
Linear (FC)              256 × 58            14,906
Bias terms               —                    ~4,816
─────────────────────────────────────────────────────
TOTAL                                        840,058
```

### Stacked LSTM Diagram

```
Input Sequence (30 chars)
        │
        ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐
   │  Embed  │   │  Embed  │   │  Embed  │  ...  │  Embed  │
   │  char 1 │   │  char 2 │   │  char 3 │       │  char 30│
   └────┬────┘   └────┬────┘   └────┬────┘       └────┬────┘
        │              │              │                 │
        ▼              ▼              ▼                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │                   LSTM LAYER 1                          │
   │   h₁ ──────────► h₂ ──────────► h₃ ──────► ... h₃₀   │
   └─────────────────────────────────────────────────────────┘
        │              │              │                 │
        ▼              ▼              ▼                 ▼
   ┌─────────────────────────────────────────────────────────┐
   │                   LSTM LAYER 2                          │
   │   h₁ ──────────► h₂ ──────────► h₃ ──────► ... h₃₀   │
   └─────────────────────────────────────────────────────────┘
                                                       │
                                               Take last output
                                                       │
                                                       ▼
                                              Linear(256 → 58)
                                                       │
                                                       ▼
                                            Next character logits
```

---

## 💻 The Code

```python
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

    if (epoch + 1) % 50 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {current_lr:.5f}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')

print("-" * 50)
print(f"Training complete! Best loss: {best_loss:.4f}")

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
```

---

## 📊 Training Diagrams & Results

### Loss Curve

```
Loss
 │
0.80 ┤█
     │ █
0.70 ┤  █
     │   █
0.60 ┤    █
     │     █
0.50 ┤      ██
     │        █
0.40 ┤         ██
     │           █
0.30 ┤            ██
     │  LR: 0.002   █◄── LR halved to 0.001
0.20 ┤               ██
     │  LR: 0.001      █◄── LR halved to 0.0005
0.15 ┤                  ████
     │
     └──────────────────────────────────────────► Epochs
     0    50   100   150   200   250   300

  Epoch  50:  Loss = 0.7613   LR = 0.00200
  Epoch 100:  Loss = 0.5014   LR = 0.00200
  Epoch 150:  Loss = 0.4603   LR = 0.00200
  Epoch 200:  Loss = 0.2681   LR = 0.00100  ← scheduler triggered
  Epoch 250:  Loss = 0.2069   LR = 0.00050  ← scheduler triggered again
  Epoch 300:  Loss = 0.1565   LR = 0.00050
  Best loss:  0.1490
```

### What Loss Means

```
Loss Value    Model Understanding
──────────────────────────────────────────────────────────
~4.06         Random guessing  (log(58) = 4.06)
~2.5          Learned basic letter frequencies
~1.5          Learned common words and spaces
~0.8          Learning sentence structure
~0.5          Learning dialogue patterns
~0.15  ✅     Strong memorization of training text
```

### Learning Rate Schedule

```
LR
  │
0.002 ┼────────────────────────────┐
      │                            │ plateau detected
0.001 ┼                            └──────────┐
      │                                       │ plateau detected
0.0005┼                                       └──────────────────
      │
      └──────────────────────────────────────────────────► Epochs
      0          100         200         300

  ReduceLROnPlateau: if loss doesn't improve for 10 epochs → LR × 0.5
```

### Mini-Batch Training Flow

```
Full Dataset (19,970 sequences)
        │
        ▼
  Shuffle randomly each epoch
        │
        ▼
  Split into batches of 128
  ┌──────────┐ ┌──────────┐ ┌──────────┐       ┌──────────┐
  │ Batch 1  │ │ Batch 2  │ │ Batch 3  │  ...  │ Batch 157│
  │ 128 seqs │ │ 128 seqs │ │ 128 seqs │       │ 128 seqs │
  └────┬─────┘ └────┬─────┘ └────┬─────┘       └────┬─────┘
       │             │             │                  │
    forward       forward       forward            forward
    backward      backward      backward           backward
    update        update        update             update
       │
       ▼
  157 updates per epoch × 300 epochs = 47,100 total weight updates
```

### Temperature Sampling Diagram

```
Model Output (logits for 58 chars):
  'a': 2.1,  'b': 0.3,  'e': 1.8,  ' ': 3.2,  ...

After dividing by temperature:

  Temp 0.2  →  amplifies differences  →  ' ': 16.0  (almost always picked)
  ┌─────────────────────────────────────────┐
  │ ' ' ██████████████████████████████████ │ 95%
  │ 'a' █                                  │  3%
  │ 'e' ▌                                  │  2%
  └─────────────────────────────────────────┘

  Temp 1.0  →  original probabilities
  ┌─────────────────────────────────────────┐
  │ ' ' ████████████████                   │ 50%
  │ 'a' ████████                           │ 25%
  │ 'e' ██████                             │ 18%
  │ 'b' ██                                 │  7%
  └─────────────────────────────────────────┘

  Temp 1.5  →  flattens differences  →  more surprising picks
  ┌─────────────────────────────────────────┐
  │ ' ' ████████                           │ 35%
  │ 'a' ███████                            │ 28%
  │ 'e' ██████                             │ 22%
  │ 'b' █████                              │ 15%
  └─────────────────────────────────────────┘
```

---

## ✍️ Generated Text

### Starting Sequence
```
"First Citizen: Before we proce"   (first 30 chars of training text)
```

### Temperature 0.8 — 300 characters
```
First Citizen: Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen: Well, I'll hear it, sir: yet you must not think
to fob off our disgrace with a tale: but, an 't please you,
deliver.

MENENIUS: Sir, I shall tell you. With a kind of smile,
Which ne'er came from the lungs, but even thus--
For, look y
```

### Temperature Comparison

| Temp | Output (100 chars) | Character |
|---|---|---|
| **0.2** | `...hear me speak.\nAll:\nSpeak, speak.\nFirst Citizen: I say unto you, what he hath done` | Very safe, repetitive |
| **0.5** | `...hear me speak.\nAll:\nSpeak, speak.\nFirst Citizen: You are all resolved rather to di` | Focused |
| **1.0** | `...hear me speak.\nAll:\nSpeak, speak.\nFirst Citizen: We cannot, sir, we are undone alr` | Balanced |
| **1.5** | `...hear me speak.\nAll:\nSpeak, speak.\nFirst Citizen: Your belly's answer? What! The ki` | Creative |

### Quality Assessment

```
What the model learned:
  ✅  Speaker dialogue format  →  "First Citizen:", "MENENIUS:", "All:"
  ✅  Proper capitalization    →  Names, sentence starts
  ✅  Punctuation patterns     →  commas, colons, apostrophes, dashes
  ✅  Real English words       →  No gibberish at temp 0.8
  ✅  Sentence structure       →  Coherent clauses and conjunctions
  ⚠️  Diversity               →  All temps share the same opening (small dataset)
```

---

## 🏆 Final Performance Summary

```
┌──────────────────────────────────────────────┐
│           TRAINING SUMMARY                   │
├──────────────────────────────────────────────┤
│  Dataset        Tiny Shakespeare (20k chars) │
│  Vocabulary     58 unique characters         │
│  Sequences      19,970                       │
│  Device         CUDA (GPU)                   │
│  Parameters     840,058                      │
├──────────────────────────────────────────────┤
│  Initial Loss   0.7613  (epoch 50)           │
│  Final Loss     0.1565  (epoch 300)          │
│  Best Loss      0.1490                       │
│  Improvement    ~80% reduction               │
└──────────────────────────────────────────────┘
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install torch numpy requests

# Run the script
python model.py
```

**To improve results:**

```python
# Use full dataset (remove the slice)
text = response.text           # ~1MB instead of 20k chars

# Train longer
epochs = 1000

# Bigger model
hidden_size = 512
num_layers = 3
```

---

## 📚 References

- [Hochreiter & Schmidhuber (1997) — Original LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Karpathy's char-rnn](https://github.com/karpathy/char-rnn)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)