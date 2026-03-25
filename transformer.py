import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for market trend classification.
    Classifies the next-step trend into: 0=Bearish, 1=Neutral, 2=Bullish.
    """
    def __init__(self, input_dim=5, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head: Bearish (0), Neutral (1), Bullish (2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = self.input_linear(src)          # -> (batch, seq, d_model)
        src = src.transpose(0, 1)             # -> (seq, batch, d_model) for PosEncoder
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)             # -> (batch, seq, d_model)

        output = self.transformer_encoder(src)

        # Use the last time step's representation for classification
        last_step = output[:, -1, :]
        return self.classifier(last_step)


def create_sequences(data, seq_length=30):
    """
    Convert (N, features) array to labelled sequences.
    Label is determined by whether the next close is up/down/flat vs current close.
    
    Args:
        data: np.ndarray of shape (N, features). Close price assumed at index 3.
        seq_length: int, length of each input sequence.
    
    Returns:
        X: np.ndarray of shape (num_seq, seq_length, features)
        y: np.ndarray of shape (num_seq,) with labels {0, 1, 2}
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        current_close = data[i + seq_length - 1, 3]  # Close is at index 3
        next_close = data[i + seq_length, 3]

        if current_close == 0:
            continue

        ret = (next_close - current_close) / current_close
        if ret > 0.001:
            y = 2  # Bullish
        elif ret < -0.001:
            y = 0  # Bearish
        else:
            y = 1  # Neutral

        xs.append(x)
        ys.append(y)

    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)
