# Character-level RNN in PyTorch

This project implements an Advanced Character-level Recurrent Neural Network (RNN) using PyTorch. It is designed to learn character sequences from a sample text and generate new text based on the learned patterns.

## Project Overview

The model is trained on a specific string: `"Hello everyone my name is Rohit."` It uses a sliding window approach (sequencing) to create training samples where the model learns to predict the next character given a sequence of preceding characters.

### Key Features
- **Custom RNN Architecture**: Includes an embedding layer, a hidden RNN layer, and a fully connected output layer.
- **Sequence Processing**: Converts raw text into numerical indices for training.
- **Text Generation**: Features a sampling mechanism with "temperature" control to balance between deterministic and creative text generation.
- **Training Loop**: Uses the Adam optimizer and Cross Entropy Loss.

## Technical Details

- **Language**: Python 3.12+
- **Framework**: PyTorch
- **Core Components**:
  - `Embedding Layer`: Maps character indices to dense vectors.
  - `RNN Layer`: Processes sequences and maintains hidden state.
  - `Linear Layer`: Maps hidden states back to vocabulary size for prediction.

### Hyperparameters
- `seq_length`: 5 (length of input sequence)
- `embedding_dim`: 16
- `hidden_size`: 128
- `learning_rate`: 0.01
- `epochs`: 150

## Project Structure

- `main.py`: The primary script containing data preparation, model definition, training, and text generation logic.
- `pyproject.toml` / `uv.lock`: Dependency management files (using `uv`).
- `requirements.txt`: Standard pip requirements file.

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Model**:
   ```bash
   python main.py
   ```

The script will train the model for 150 epochs and then attempt to generate 50 characters of text starting from the seed "Hello".
