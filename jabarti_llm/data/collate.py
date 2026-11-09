

import torch


def collate_fn(examples, pad_id=0):
    

    chunks = [ example["input_ids"] for example in examples ]
    longest = max( len(chunk) for chunk in chunks)

    padded = torch.full(
        (len(chunks), longest), pad_id, dtype=torch.long
    )
    for ix, chunk in enumerate(chunks):
        padded[ix, :len(chunk)] = chunk

    return {
        "input_ids": padded[: , :-1],
        "targets": padded[:, 1:]
    }
