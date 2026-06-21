from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.rnn(emb)
        return h  # (1, N, H)


class SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward_step(self, token, hidden):
        # token: (N,) long
        emb = self.embed(token).unsqueeze(1)  # (N,1,E)
        out, hidden = self.rnn(emb, hidden)
        logits = self.out(out.squeeze(1))
        return logits, hidden


def greedy_decode(decoder: SeqDecoder, start_token: int, hidden: torch.Tensor, max_len: int = 20) -> List[int]:
    device = hidden.device
    token = torch.tensor([start_token], dtype=torch.long, device=device)
    seq = []
    for _ in range(max_len):
        logits, hidden = decoder.forward_step(token, hidden)
        token = logits.argmax(dim=1)
        t = int(token.item())
        seq.append(t)
        if t == 0:
            break
    return seq


def beam_search_decode(decoder: SeqDecoder, start_token: int, hidden: torch.Tensor, beam_width: int = 3, max_len: int = 20) -> List[int]:
    device = hidden.device
    sequences: List[Tuple[List[int], float, torch.Tensor]] = [([], 0.0, hidden)]
    for _ in range(max_len):
        all_candidates = []
        for seq, score, h in sequences:
            token = torch.tensor([seq[-1]] if seq else [start_token], dtype=torch.long, device=device)
            logits, h_new = decoder.forward_step(token, h)
            logp = F.log_softmax(logits, dim=1).squeeze(0)
            topk = torch.topk(logp, beam_width)
            for k in range(beam_width):
                idx = int(topk.indices[k].item())
                sc = score + float(topk.values[k].item())
                all_candidates.append((seq + [idx], sc, h_new))
        # select best beam_width
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    # return best sequence
    return sequences[0][0]


if __name__ == '__main__':
    # tiny smoke test
    vocab = 50
    enc = SeqEncoder(vocab)
    dec = SeqDecoder(vocab)
    x = torch.randint(2, vocab, (1, 5))
    h = enc(x)
    print('encoded', h.shape)
    print('greedy', greedy_decode(dec, start_token=2, hidden=h))
    print('beam', beam_search_decode(dec, start_token=2, hidden=h))
