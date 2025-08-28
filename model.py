import torch
import math
import torch.nn as nn
from config import MAX_NUM_POINTS 


# --- Part 1: Positional Encoding ---
# This is a standard module to inject position information into the input.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# --- Part 2: The Transformer Network ---
class TransformerEmbeddingNet(nn.Module):
    def __init__(
        self, input_dim=2, d_model=64, nhead=4, d_hid=128, nlayers=3, dropout=0.0
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Input Projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_NUM_POINTS)

        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        # 4. Final normalization layer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]

        Returns:
            Tensor, shape [batch_size, d_model] (the summary vector)
        """
        # 1. Create the padding mask.
        # Our padded time is -2, so we can use that to create the mask.
        src_key_padding_mask = src[:, :, 0] == -2.0

        # 2. Project input, apply scaling and positional encoding
        src = self.input_proj(src) * math.sqrt(self.d_model)

        # PyTorch's TransformerEncoderLayer expects batch_first=True, but the
        # PositionalEncoding and the container TransformerEncoder do not have this
        # argument and expect (seq_len, batch, dim). So we permute.
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)

        # 3. Pass through the transformer encoder
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # 4. Aggregate the output. 

        # # 1. Build a float mask that is 1.0 for real data, 0.0 for pads
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()   # (B, L, 1)
        # 2. Zero-out the padded positions, then sum and normalise
        masked_output = output * valid_mask                          # (B, L, d_model)
        embedding_sum = masked_output.sum(dim=1)                     # (B, d_model)
        # number of real time-steps per example (avoid division by zero with clamp)
        lengths = valid_mask.sum(dim=1).clamp(min=1.0)               # (B, 1)
        embedding = embedding_sum / lengths                          # final pooled vector

        # 5. Apply a final layer norm for stability
        embedding = self.layer_norm(embedding)

        return embedding
