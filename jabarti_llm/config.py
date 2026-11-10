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
    pad_id:int = 0 # padding token id
    tie_weights: bool = True # tie input and output embeddings
    
    @property
    def d_k(self):
        return self.d_model // self.n_heads
    @classmethod
    def jabarti(cls):
        return cls(
            d_model=512, n_heads=8, n_layers=8,
            dropout=0.05
        )

    @classmethod
    def tiny(cls):
        return cls(
            d_model=128, n_heads=4, n_layers=2, max_seq_len=256
        )


@dataclass
class GenerationConfig:


    max_new_tokens: int = 100
    temperature: float = 0.8

    top_k: int | None = 50     
    top_p: float | None = 0.95   
                                  
    use_cache: bool = True      
    seed: int | None = None


@classmethod
def greedy(cls):
        """Always take the likeliest token. Deterministic, and repetitive."""
        return cls(temperature=0.0, top_k=None, top_p=None)