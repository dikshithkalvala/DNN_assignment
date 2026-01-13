# src/utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import math
import time
import os


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device=None):
    device = device or torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


# Simple BLEU-4 (corpus-level, short)
def bleu4_score(references: List[str], hypotheses: List[str]) -> float:
    """
    Very small approximation: geometric mean of n-gram precisions with brevity penalty.
    references, hypotheses: lists of strings (tokenized by whitespace)
    """
    import collections

    def ngram_counts(s, n):
        toks = s.split()
        return collections.Counter([tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))])

    precisions = []
    for n in range(1, 5):
        num = 0
        den = 0
        for ref, hyp in zip(references, hypotheses):
            hc = ngram_counts(hyp, n)
            rc = ngram_counts(ref, n)
            den += sum(hc.values())
            # clipped count
            clip = sum(min(hc[ng], rc.get(ng, 0)) for ng in hc)
            num += clip
        p = (num / den) if den > 0 else 0.0
        precisions.append(p if p > 0 else 1e-9)
    # geometric mean
    score = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** (1 / 4)
    # brevity penalty
    ref_len = sum(len(r.split()) for r in references)
    hyp_len = sum(len(h.split()) for h in hypotheses)
    bp = math.exp(1 - ref_len / hyp_len) if hyp_len < ref_len and hyp_len > 0 else 1.0
    return bp * score


# ROUGE-L via LCS
def rouge_l_score(reference: str, hypothesis: str) -> float:
    a = reference.split()
    b = hypothesis.split()
    # LCS length
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    if lcs == 0:
        return 0.0
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    f = (2 * prec * rec) / (prec + rec)
    return f


# training loop (minimal)
def train_epoch(model, dataloader, optimizer, device, vocab, clip=1.0):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad idx = 0
    for batch in dataloader:
        images = batch["images"].to(device)  # (B,S,C,H,W)
        texts = batch["texts"].to(device)  # (B,S,L)
        target_texts = batch["target_text"].to(device)  # (B,L)
        # forward
        logits, img_out = model(images, texts, target_texts)
        # logits: (B,L,V)
        B, L, V = logits.shape
        loss = criterion(logits.view(B * L, V), target_texts.view(B * L))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    refs = []
    hyps = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            texts = batch["texts"].to(device)
            target_texts = batch["target_text"].to(device)
            raw_targets = batch["raw_target_text"]
            logits, img_out = model(images, texts, target_texts=None)  # generation mode
            # logits may be generated ids if model implements that; here StoryModel returns ids when target_texts None
            # If logits is tensor of ids:
            if isinstance(logits, torch.Tensor) and logits.dtype == torch.long:
                preds = logits.cpu().tolist()
                for p in preds:
                    # convert ids to tokens if vocab present
                    toks = []
                    for idx in p:
                        toks.append(vocab.itos[idx] if idx < len(vocab.itos) else "<unk>")
                    hyps.append(" ".join([t for t in toks if t not in ("<pad>", "<bos>", "<eos>")]))
            else:
                # if logits are probabilities (teacher forcing shape), take argmax
                B, L, V = logits.shape
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                for p in pred_ids:
                    toks = []
                    for idx in p:
                        toks.append(vocab.itos[idx] if idx < len(vocab.itos) else "<unk>")
                    hyps.append(" ".join([t for t in toks if t not in ("<pad>", "<bos>", "<eos>")]))

            # refs
            refs.extend(raw_targets)

    # compute simple BLEU and ROUGE-L
    bleu = bleu4_score(refs, hyps)
    rouge_l = sum(rouge_l_score(r, h) for r, h in zip(refs, hyps)) / max(1, len(refs))
    return {"bleu4": bleu, "rouge_l": rouge_l, "n_samples": len(refs)}


# Example create optimizer
def make_optimizer(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr=lr)
