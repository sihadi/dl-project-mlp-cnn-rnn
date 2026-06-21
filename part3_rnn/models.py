import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = h[-1]
        return self.fc(h)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.rnn(emb)
        h = h[-1]
        return self.fc(h)


class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        h = h[-1]
        return self.fc(h)
