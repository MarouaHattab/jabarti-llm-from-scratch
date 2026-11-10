"""
this script shows how the learning rate changes over a run
"""

import math

import torch
def scale_for_step(step,config):
        
        # === warmup
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps

        # === cosine decay
        decay_steps = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, (step - config.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # === floor
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

def cosine_with_warmup(optimizer, config):
     return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scale_for_step(step, config)
    )