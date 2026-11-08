from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int =32_000
    d_model: int = 768 # Embedding dimension 
    d_ff :int = 3072 # Feedforward dimension 4*768 GPT 2's ratio
    max_seq_len: int =1024 
    dropout: float = 0.1 # to avoid overfitting
    n_layers: int = 12 # number of transformer blocks
    n_heads: int = 12 # number of attention heads
    qkv_bias : bool =False 
    
    @property
    def d_k(self):
        return self.d_model // self.n_heads
    