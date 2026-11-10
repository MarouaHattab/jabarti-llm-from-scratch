import torch
import torch.nn.functional as F


def greedy(logits):
    return int(torch.argmax(logits).item())


def apply_temperature(logits, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy() for temperature 0")

    return logits / temperature


def top_k_filter(logits, k):
    if k is None or k >= logits.size(-1):
        return logits

    kth_best = torch.topk(logits, k).values[-1]

    # logits = [3.0,  1.0,   2.0,   0.5,   -1.0]
    #          tok0   tok1   tok2   tok3   tok4

    # torch.topk(logits, 3).values  →  [3.0, 2.0, 1.0]     sorted, biggest first
    #                                  [-1]  →   1.0       the smallest of the winners

    # before :  [3.0,  1.0,  2.0,  0.5, -1.0]
    # after  :  [3.0,  1.0,  2.0, -inf, -inf]

    return logits.masked_fill(logits < kth_best, float("-inf"))




def top_p_filter(logits, p):

    if p is None or p >= 1.0:
        return logits

    # logits =  [1.0, 3.0, 0.0, 2.0, -1.0]     p = 0.9
    #           tok0 tok1 tok2 tok3 tok4

    # ordered  : [3.0, 2.0, 1.0, 0.0, -1.0]
    # indices  : [ 1,   3,   0,   2,   4 ]     ← tok1 was biggest, tok3 second…

    ordered, indices = torch.sort(logits, descending=True)

    # probs      : [0.636, 0.234, 0.086, 0.032, 0.012]    ← individual
    probs = F.softmax(ordered, dim=-1)

    # cumulative : [0.636, 0.871, 0.957, 0.988, 1.000]    ← running sum
    #                             ↑
    #                             first to pass p = 0.9
    cumulative = torch.cumsum(probs, dim=-1)

    # which to remove ?
    removed = cumulative > p # [F, F, T, T, T]
    removed[1:] = removed[:-1].clone()
    removed[0] = False

    # removed = [F, F, F, T, T]
    filtered = ordered.masked_fill(removed, float("-inf"))

    # put them back in token order

    # out : [1.0, 3.0, -inf, 2.0, -inf]     back to tok0..tok4
    #     → [0.09, 0.665, 0.0, 0.245, 0.0]

    # out.scatter_(-1, indices, filtered)
    # means:
    # for j in range(len(filtered)):
    #   out[ indices[j] ] = filtered[j]

    return torch.empty_like(logits).scatter_(-1, indices, filtered)




def sample(logits, config, generator=None):
    if config.temperature == 0:
        return greedy(logits)

    logits = apply_temperature(logits, config.temperature)
    logits = top_k_filter(logits, config.top_k)
    logits = top_p_filter(logits, config.top_p)

    probabilities = F.softmax(logits, dim=-1)

    return int(
          torch.multinomial(
              probabilities,
              num_samples=1,
              generator=generator).item()
        )