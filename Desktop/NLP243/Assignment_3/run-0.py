import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import numpy as np

# SubwordDataset class remains unchanged
class SubwordDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long)

# Custom collate function for dynamic padding
def collate_fn(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch[:, :-1], batch[:, 1:]

# Transformer with attention
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.transformer(x.permute(1, 0, 2))
        x = self.fc_out(x.permute(1, 0, 2))
        return x

# Load PTB dataset and preprocess
def load_ptb_dataset(max_vocab_size=20000, max_len=50):
    tokenizer = get_tokenizer("basic_english")

    train_iter = PennTreebank(split="train")
    valid_iter = PennTreebank(split="valid")
    test_iter = PennTreebank(split="test")

    train_data = [tokenizer(line) for line in train_iter]
    valid_data = [tokenizer(line) for line in valid_iter]
    test_data = [tokenizer(line) for line in test_iter]

    counter = Counter([token for line in train_data for token in line])
    vocab = Vocab(counter, max_size=max_vocab_size)

    def encode_data(data):
        return [[vocab[token] for token in line if token in vocab] for line in data]

    train_encoded = encode_data(train_data)
    valid_encoded = encode_data(valid_data)
    test_encoded = encode_data(test_data)

    train_dataset = SubwordDataset(train_encoded, max_len)
    valid_dataset = SubwordDataset(valid_encoded, max_len)
    test_dataset = SubwordDataset(test_encoded, max_len)

    return train_dataset, valid_dataset, test_dataset, len(vocab)

# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.view(-1, output.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = model(x_val)
                val_loss = criterion(val_output.view(-1, val_output.size(-1)), y_val.view(-1))
                total_val_loss += val_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {total_val_loss / len(valid_loader):.4f}")

# Generate submission.csv
def generate_submission(model, test_loader, vocab_size, device, output_file="submission.csv"):
    model.eval()
    model.to(device)
    predictions = []

    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            predicted_tokens = output.argmax(dim=-1)
            predictions.extend(predicted_tokens.cpu().numpy())

    np.savetxt(output_file, predictions, delimiter=",", fmt="%d")
    print(f"Submission file saved as {output_file}")

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    max_len = 50
    embed_size = 128
    num_heads = 4
    num_layers = 6
    batch_size = 120
    num_epochs =2
    learning_rate = 1e-3

    # Load dataset
    train_dataset, valid_dataset, test_dataset, vocab_size = load_ptb_dataset(max_vocab_size=20000, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, loss, and optimizer
    model = TransformerModel(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers, max_len=max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Generate predictions
    generate_submission(model, test_loader, vocab_size, device)
