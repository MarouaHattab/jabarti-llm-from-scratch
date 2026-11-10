
def freeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    

def unfreeze_tail(model, n_blocks: int = 2, unfreeze_lm_head: bool = True):
    for block in model.blocks[-n_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.ln_final.parameters():
        parameter.requires_grad = True

    if unfreeze_lm_head:
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True

    