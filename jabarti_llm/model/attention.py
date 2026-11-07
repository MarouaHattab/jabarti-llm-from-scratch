import torch 
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    # Input (batch_size, seq_len, d_model)
    # output (batch_size, seq_len, d_model)
    def __init__(self,config):
        super().__init__()
        self.d_k = config.d_k
        self.n_heads = config.n_heads
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.W_o = nn.Linear(config.d_model, config.d_model)
        mask=torch.tril(
            torch.ones(config.max_seq_len,config.max_seq_len,dtype=torch.bool)
        )
        self.register_buffer("mask",mask,persistent=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = q @ k.transpose(-2, -1)
        scores = scores / self.d_k**0.5
        
        scores = scores.masked_fill(~self.mask, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.W_o(out), weights
                