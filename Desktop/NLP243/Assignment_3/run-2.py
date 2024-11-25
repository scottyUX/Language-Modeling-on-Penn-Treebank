import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


# Positional Encoding
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        pos = torch.arange(x.size(1), device=x.device).view(1, x.size(1))  # (1, seq_len)
        embedding = self.pos_embedding(pos)  # (1, seq_len, d_model)
        return x + embedding


# Transformer-based Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)

        # Output layer to map to vocab size
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: (batch, seq_len)
        # Clamp indices to avoid out-of-range errors
        x = x.clamp(0, self.embedding.num_embeddings - 1)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)  # (seq_len, seq_len)
        x = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)

        logits = self.output_layer(x)  # (batch, seq_len, vocab_size)
        return logits


# Dataset Preparation
class SubwordDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.tokenizer = tokenizer
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded = self.tokenizer.encode(sentence)
        inputs = encoded.ids[:-1]
        targets = encoded.ids[1:]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def build_tokenizer(sentences, vocab_size=5000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(special_tokens=["<PAD>", "<UNK>"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>")
    tokenizer.enable_truncation(max_length=50)
    return tokenizer


def preprocess_data(vocab_size=5000):
    ptb = load_dataset('ptb_text_only')
    train_sentences = [item['sentence'] for item in ptb['train']]
    val_sentences = [item['sentence'] for item in ptb['validation']]
    test_sentences = [item['sentence'] for item in ptb['test']]

    tokenizer = build_tokenizer(train_sentences, vocab_size=vocab_size)

    train_dataset = SubwordDataset(train_sentences, tokenizer)
    val_dataset = SubwordDataset(val_sentences, tokenizer)
    test_dataset = SubwordDataset(test_sentences, tokenizer)

    print(f"New Vocabulary Size: {tokenizer.get_vocab_size()}")
    return train_dataset, val_dataset, test_dataset, tokenizer


# Custom collate function for dynamic padding
def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    padded_inputs = [torch.cat([seq, torch.tensor([0] * (max_len - len(seq)))]) for seq in inputs]
    padded_targets = [torch.cat([seq, torch.tensor([0] * (max_len - len(seq)))]) for seq in targets]
    return torch.stack(padded_inputs).long(), torch.stack(padded_targets).long()


# Training Loop
def train_model(model, train_loader, val_loader, vocab_size, epochs=2, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> token
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:  # Log every 10 batches
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}")


# Generate submission.csv
def generate_submission(model, test_loader, vocab_size, output_file, device):
    model.eval()
    perplexities = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            perplexity = torch.exp(loss / targets.numel())
            perplexities.append(perplexity.item())

    submission_df = pd.DataFrame({"ID": range(len(perplexities)), "ppl": perplexities})
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")


# Main Script
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    # Update the vocabulary size to 20,000
    train_dataset, val_dataset, test_dataset, tokenizer = preprocess_data(vocab_size=20000)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    vocab_size = tokenizer.get_vocab_size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Update the number of layers to 6
    model = TransformerLM(vocab_size, d_model=128, n_head=4, n_layer=6).to(device)

    train_model(model, train_loader, val_loader, vocab_size, epochs=5)
    generate_submission(model, test_loader, vocab_size, output_file, device)
