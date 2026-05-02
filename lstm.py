# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("/kaggle/input/datasets/abdulbasit1287/cryptosentimentanalysisds1-0/cryptoSentimentAnalysisDS1.0.csv")

# combine text
df["text"] = df["headline"].astype(str) + " " + df["content"].astype(str)

# clean labels
df = df[df["sentiment"].notna()]
df["sentiment"] = df["sentiment"].astype(int)

# label mapping (safe)
label_map = {v: i for i, v in enumerate(sorted(df["sentiment"].unique()))}
df["sentiment"] = df["sentiment"].map(label_map)

print("Label mapping:", label_map)

texts = df["text"].values
labels = df["sentiment"].values

num_classes = len(label_map)

# =========================
# 3. TOKENIZATION
# =========================
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

tokenized_texts = [tokenize(t) for t in texts]

# =========================
# 4. VOCAB BUILD
# =========================
all_words = [w for sent in tokenized_texts for w in sent]
counter = Counter(all_words)

vocab = {"<PAD>": 0, "<UNK>": 1}
for w, c in counter.items():
    if c > 1:
        vocab[w] = len(vocab)

vocab_size = len(vocab)

# =========================
# 5. ENCODING
# =========================
MAX_LEN = 40

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(w, 1) for w in tokens]
    ids = ids[:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    return ids

X = np.array([encode(t) for t in texts])
y = np.array(labels)

# =========================
# 6. DATASET CLASS
# =========================
class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CryptoDataset(X, y)

# =========================
# 7. TRAIN / TEST SPLIT
# =========================
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# =========================
# 8. LSTM MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        lstm_out, (h, _) = self.lstm(x)

        avg_pool = torch.mean(lstm_out, dim=1)
        last_hidden = torch.cat((h[-2], h[-1]), dim=1)

        out = avg_pool + last_hidden
        out = self.dropout(out)

        return self.fc(out)

# =========================
# 9. INIT MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    vocab_size=vocab_size,
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 10. TRAINING
# =========================
epochs = 5

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# =========================
# 11. TEST EVALUATION
# =========================
model.eval()

correct, total = 0, 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print("Test Accuracy:", correct / total)

# =========================
# 12. PLOTS (IMPORTANT)
# =========================

# Accuracy plot
plt.figure()
plt.plot(range(1, epochs + 1), train_accuracies)
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Loss plot
plt.figure()
plt.plot(range(1, epochs + 1), train_losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
