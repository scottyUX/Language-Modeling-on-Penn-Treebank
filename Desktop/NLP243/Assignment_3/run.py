import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

# Load a small subset of the Penn Treebank dataset for quick testing
ptb = load_dataset("ptb_text_only")

# Extract text samples
train_texts = [entry["sentence"] for entry in ptb["train"]]
val_texts = [entry["sentence"] for entry in ptb["validation"]]
test_texts = [entry["sentence"] for entry in ptb["test"]]

# # Limit for testing
# train_texts = train_texts[:500]  # First 100 samples for training
# val_texts = val_texts[500:550]  # First 50 samples for validation
# test_texts = test_texts[550:600]  # First 50 samples for testing

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx], 
            truncation=True, 
            padding='max_length', 
            max_length=self.seq_len, 
            return_tensors="pt"
        )
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0)

# Transformer Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, seq_len):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).to(x.device).bool()
        output = self.transformer(embedded.permute(1, 0, 2), mask=mask)
        output = self.fc(output.permute(1, 0, 2))
        return output

# Hyperparameters for quick testing
seq_len = 128
batch_size = 64
epochs = 5
learning_rate = 1e-3
d_model = 32
n_head = 8
n_layer = 6

# DataLoaders
train_dataset = TextDataset(train_texts, tokenizer, seq_len)
val_dataset = TextDataset(val_texts, tokenizer, seq_len)
test_dataset = TextDataset(test_texts, tokenizer, seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model, Optimizer, and Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerLM(vocab_size, d_model, n_head, n_layer, seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training function
def train_model(model, train_loader, val_loader, device, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            inputs = inputs.to(device)
            targets = inputs.clone()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, perplexity = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
    torch.save(model.state_dict(), "transformer_lm_test.pth")
    print("Model saved as transformer_lm_test.pth")

# Evaluation function
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = inputs.clone()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

# Submission function
# def generate_submission(model, test_loader, vocab_size, output_file, device):
#     model.eval()
#     perplexities = []
#     with torch.no_grad():
#         for inputs, _ in tqdm(test_loader, desc="Generating Predictions"):
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
#             loss = criterion(outputs.view(-1, vocab_size), inputs.view(-1))
#             perplexity = torch.exp(loss / inputs.numel())
#             perplexities.append(perplexity.item())
#     submission = pd.DataFrame({"ID": range(len(perplexities)), "ppl": perplexities})
#     submission.to_csv(output_file, index=False)
#     print(f"Submission file saved to {output_file}")
def generate_submission(model, test_loader, vocab_size, output_file, device):
    model.eval()
    perplexities = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Generating Predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Calculate loss for each sequence
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            loss = criterion(outputs.view(-1, vocab_size), inputs.view(-1))
            loss = loss.view(inputs.size(0), -1).sum(dim=1)  # Sum loss per sequence
            perplexity = torch.exp(loss / inputs.size(1))    # Sequence-level perplexity
            perplexities.extend(perplexity.cpu().numpy().tolist())

    # Create DataFrame with all sequences
    submission = pd.DataFrame({"ID": range(len(perplexities)), "ppl": perplexities})
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")


# Run quick test
output_file = "submission.csv"
train_model(model, train_loader, val_loader, device, optimizer, criterion, epochs)
generate_submission(model, test_loader, vocab_size, output_file, device)
