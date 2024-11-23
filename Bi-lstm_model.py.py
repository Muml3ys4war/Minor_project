import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import threading

# Step 1: Load GloVe embeddings
def load_glove_embeddings(glove_file, embedding_dim):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load pre-trained GloVe embeddings
embedding_dim = 50
glove_file = './.vector_cache/glove.6B.50d.txt'
glove_embeddings = load_glove_embeddings(glove_file, embedding_dim)

# Step 2: Load and preprocess data
file_path = './IMDB Dataset.csv'
df = pd.read_csv(file_path)
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])  # Encode labels

# Convert text to TF-IDF features and apply Chi-square
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['review'])
y = df['sentiment']

k_best_features = 1000
chi2_selector = SelectKBest(chi2, k=k_best_features)
X_kbest = chi2_selector.fit_transform(X_tfidf, y)

# Build vocabulary of selected features
selected_feature_names = tfidf.get_feature_names_out()[chi2_selector.get_support()]
selected_vocab = {word: idx for idx, word in enumerate(selected_feature_names)}

# Create embedding matrix for selected vocabulary
def build_embedding_matrix(vocab, glove_embeddings, embedding_dim):
    matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        vector = glove_embeddings.get(word)
        if vector is not None:
            matrix[idx] = vector
        else:
            matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(matrix, dtype=torch.float)

embedding_matrix = build_embedding_matrix(selected_vocab, glove_embeddings, embedding_dim)

# Step 3: Define Dataset and DataLoader
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_indices = torch.tensor([self.vocab.get(word, 0) for word in text.split()], dtype=torch.long)
        return text_indices, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

# Step 4: Define BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout_prob, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step output
        output = self.fc(lstm_out)
        return output

# Model parameters
hidden_dim = 128
output_dim = len(label_encoder.classes_)
num_layers = 2
dropout_prob = 0.3
vocab_size = len(selected_vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_matrix = embedding_matrix.to(device)

# Step 5: Train and evaluate model using K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
num_epochs = 2
batch_size = 128

fold_accuracies = []
threads = []

def train_and_evaluate_model(train_indices, val_indices, fold_num):
    X = df['review'].tolist()
    y = df['sentiment'].tolist()

    # Split data into training and validation sets
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]

    # Create DataLoaders
    train_dataset = SentimentDataset(X_train, y_train, selected_vocab)
    val_dataset = SentimentDataset(X_val, y_val, selected_vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = BiLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout_prob, embedding_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Fold {fold_num} - Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predictions = torch.max(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = accuracy_score(targets, preds)
    print(f"Fold {fold_num} - Validation Accuracy: {accuracy:.4f}")
    fold_accuracies.append((fold_num, accuracy))
    torch.save(model.state_dict(), "bilstm_model.pth")

# Launch threads for each fold
for fold_num, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
    thread = threading.Thread(target=train_and_evaluate_model, args=(train_idx, val_idx, fold_num))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Average accuracy
average_accuracy = np.mean([acc for _, acc in fold_accuracies])
print(f"Average Accuracy across 10 folds: {average_accuracy:.4f}")

