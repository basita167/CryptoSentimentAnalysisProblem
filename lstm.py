# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("/kaggle/input/datasets/abdulbasit1287/cryptosentimentanalysisds1-0/cryptoSentimentAnalysisDS1.0.csv")

df["text"] = df["headline"].astype(str) + " " + df["content"].astype(str)

df = df[df["sentiment"].notna()]
df["sentiment"] = df["sentiment"].astype(int)

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
# 4. VOCAB
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
# 6. DATASET
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
# 7. SPLIT
# =========================
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# =========================
# 8. MODEL
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
# 9. INIT
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(vocab_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 🔥 CHECKPOINT PATH
# =========================
checkpoint_path = "lstm_checkpoint.pth"

start_epoch = 0
train_losses = []
train_accuracies = []

# =========================
# 🔥 LOAD CHECKPOINT IF EXISTS
# =========================
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["losses"]
    train_accuracies = checkpoint["accuracies"]

    print(f"Resuming from epoch {start_epoch}")

# =========================
# 10. TRAINING
# =========================
epochs = 5

for epoch in range(start_epoch, epochs):
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
    # SAVE CHECKPOINT
    # =========================
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "losses": train_losses,
        "accuracies": train_accuracies
    }, checkpoint_path)

# =========================
# 11. TEST
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
# 12. PLOTS
# =========================
plt.figure()
plt.plot(train_accuracies)
plt.title("Training Accuracy")
plt.show()

plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.show()
