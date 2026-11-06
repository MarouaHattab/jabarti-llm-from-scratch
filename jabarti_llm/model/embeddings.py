#  token IDs to the vectors the model actually reads


import torch
import torch.nn as nn



class InputEmbedding(nn.Module):
    def __init__(self, config):
       super().__init__()
       self.max_seq_len = config.max_seq_len
       self.dropout = nn.Dropout(config.dropout)
       self.token_embeddings = nn.Embedding(
        config.vocab_size, config.d_model)

       self.position_embedding = nn.Embedding(
        config.max_seq_len, config.d_model)
       

    def forward(self, input_ids,position_offset: int=0):
        _,seq_len=input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence Lenght {position_offset +seq_len} exceeds the maximum allowed {self.max_seq_len}"

            )
        positions = torch.arange(
            position_offset  , position_offset + seq_len,device=input_ids.device)
        tokens=self.token_embeddings(input_ids)
        pos=self.position_embedding(positions)

        return self.dropout(tokens + pos) # shape (batch_size, seq_len, d_model)
