import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jabarti_llm.model.block import TransformerBlock
from jabarti_llm.model.embeddings import InputEmbedding


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = InputEmbedding(config)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        scale = math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.W_o.weight.div_(scale)
                block.ffn.fc2.weight.div_(scale)

    def forward(self, input_ids, targets=None, kv_cache=None, use_cache=False):
        past_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
        x = self.embedding(input_ids, position_offset=past_len)

        updated_caches = [] if use_cache else None
        for ix, block in enumerate(self.blocks):
            current_cache = kv_cache[ix] if kv_cache is not None else None
            x, updated_cache = block(
                x, use_cache=use_cache, kv_cache=current_cache
            )
            if use_cache:
                updated_caches.append(updated_cache)

        logits = self.lm_head(self.ln_final(x))
        if targets is None:
            return logits, None, updated_caches

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.config.pad_id,
        )
        return logits, loss, updated_caches

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.embedding.position_embedding.weight.numel()
            total -= self.embedding.token_embedding.weight.numel()
        return total
