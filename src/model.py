import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch, seq_len)
        # src_key_padding_mask: (batch, seq_len) - True indicates padding
        tok = self.token_embed(src)  # (batch, seq_len, d_model)
        tok = tok + self.pos_encoder(tok)  # (batch, seq_len, d_model)
        return self.encoder(tok, src_key_padding_mask=src_key_padding_mask)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)   <- CHANGED!
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len]  # (1, seq_len, d_model), broadcasted across batch


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 26)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: (batch, seq_len)
        # memory: (batch, memory_seq_len, d_model)

        tok = self.token_embed(tgt)  # (batch, seq_len, d_model)
        tok = tok + self.pos_encoder(tok)  # (batch, seq_len, d_model)

        batch_size, seq_len, _ = tok.shape
        # Create tgt_mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()
        # Above: upper triangular mask, diagonal=1 (no future tokens allowed)

        output = self.decoder(
            tok,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (batch, seq_len, d_model)

        return self.fc_out(output)  # (batch, seq_len, 26)


class HangmanTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=64):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_layers, max_len)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers, max_len)
        self.d_model = d_model # Store d_model for mask generation
        self.max_len = max_len # Store max_len for mask generation

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: (seq_len, batch)
        # tgt: (seq_len, batch)
        # src_key_padding_mask: (batch, seq_len)
        # tgt_key_padding_mask: (batch, seq_len)

        # The memory_key_padding_mask for the decoder attending to the encoder output
        # is the same as the src_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output