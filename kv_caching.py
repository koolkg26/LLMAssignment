%%writefile train_ddp_v2.py
import os
import math
import pickle
from pathlib import Path
from functools import partial
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.nn.utils.rnn import pad_sequence
import wandb

CONTEXT_LEN = 64
BATCH_SIZE = 128
SOS_ID = 2
EOS_ID = 3
UNK_ID = 1
PAD_ID = 0
HIDDEN_DIM = 300
NUM_LAYERS = 3
NUM_HEADS = 5
DROPOUT = 0.1
LR = 3e-4
EPOCHS = 10

class ChunkedSequenceDataset(Dataset):
    def __init__(self, hf_dataset, word2idx, context_len, sos_id, eos_id, unk_id):
        self.word2idx = word2idx
        self.context_len = context_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.chunks = self._prepare_chunks(hf_dataset)

    def _tokens_to_ids(self, tokens):
        return [self.sos_id] + [self.word2idx.get(tok, self.unk_id) for tok in tokens] + [self.eos_id]

    def _prepare_chunks(self, dataset):
        all_chunks = []
        for item in tqdm(dataset, desc="Converting to chunks"):
            tokens = item.get("tokens", None)
            if not tokens:
                continue
            ids = self._tokens_to_ids(tokens)
            for i in range(0, max(1, len(ids) - 1), self.context_len + 1):
                chunk = ids[i : i + self.context_len + 1]
                if len(chunk) > 1:
                    all_chunks.append(chunk)
        return all_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)

def collate_batch(batch, pad_id, context_len):
    input_seqs, target_seqs = [], []
    for seq in batch:
        input_seq = seq[:-1][:context_len]
        target_seq = seq[1:][:context_len]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    padded_inputs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_id)
    padded_targets = pad_sequence(target_seqs, batch_first=True, padding_value=pad_id)
    if padded_inputs.size(1) < context_len:
        pad_width = context_len - padded_inputs.size(1)
        pad_tensor = torch.full((padded_inputs.size(0), pad_width), pad_id, dtype=torch.long)
        padded_inputs = torch.cat([padded_inputs, pad_tensor], dim=1)
        padded_targets = torch.cat([padded_targets, pad_tensor], dim=1)
    elif padded_inputs.size(1) > context_len:
        padded_inputs = padded_inputs[:, :context_len]
        padded_targets = padded_targets[:, :context_len]
    attention_mask = (padded_inputs != pad_id).long()
    return padded_inputs, padded_targets, attention_mask

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

def sinusoidal_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.from_numpy(pe)

import time
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, past_k=None, past_v=None, use_cache=False, output_attentions=False):
        b, t, _ = x.size()
        q = self.wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        if past_k is not None and past_v is not None:
            full_k = torch.cat([past_k, k], dim=2)
            full_v = torch.cat([past_v, v], dim=2)
            total_len = full_k.size(2)
        else:
            full_k = k
            full_v = v
            total_len = t
        scores = torch.matmul(q, full_k.transpose(-2, -1)) / self.scale
        orig_dtype = scores.dtype
        scores = scores.float()
        causal = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool, device=x.device))
        q_start = total_len - t
        causal_rows = causal[q_start: q_start + t, :]
        allowed = causal_rows.unsqueeze(0).unsqueeze(0).expand(b, self.num_heads, t, total_len)
        if padding_mask is not None:
            if past_k is not None and past_k.size(2) > 0:
                pad_bool = (padding_mask == 0) if padding_mask.dtype != torch.bool else padding_mask
                if pad_bool.dim() == 2 and pad_bool.size(1) == t:
                    prior_cols = torch.ones((b, 1, 1, total_len - t), dtype=torch.bool, device=x.device)
                    last_cols = (~pad_bool).unsqueeze(1).unsqueeze(2)
                    key_is_real_full = torch.cat([prior_cols, last_cols], dim=-1)
                else:
                    key_is_real_full = torch.ones((b,1,1,total_len), dtype=torch.bool, device=x.device)
            else:
                pad_bool = padding_mask if padding_mask.dtype == torch.bool else (padding_mask == 0)
                key_is_real_full = (~pad_bool).unsqueeze(1).unsqueeze(2)
            allowed = allowed & key_is_real_full
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn.to(full_v.dtype)
        context = torch.matmul(attn, full_v)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.wo(context)
        present_k = full_k if use_cache else None
        present_v = full_v if use_cache else None
        if output_attentions:
            return out, attn, present_k, present_v
        else:
            return out, None, present_k, present_v

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or (4 * d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim=4 * d_model, dropout=dropout)
        self.ln2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, past_k=None, past_v=None, use_cache=False, output_attentions=False):
        x_norm = self.ln1(x)
        attn_out, attn_weights, present_k, present_v = self.self_attn(
            x_norm, padding_mask=key_padding_mask, past_k=past_k, past_v=past_v, use_cache=use_cache, output_attentions=output_attentions
        )
        x = x + self.dropout(attn_out)
        x_norm2 = self.ln2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x, attn_weights, present_k, present_v

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, max_len, dropout=0.1, embedding_weights=None, freeze_embeddings=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if embedding_weights is not None:
            self.token_embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
            if freeze_embeddings:
                self.token_embedding.weight.requires_grad = False
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer("position_encoding", pe)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.final_ln = LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False):
        b, t = input_ids.size()
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_encoding[:t, :].unsqueeze(0).expand(b, -1, -1)
        x = tok_emb + pos_emb
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        all_attentions = []
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past_k = None
            layer_past_v = None
            if past_key_values is not None:
                layer_past = past_key_values[i]
                if layer_past is not None:
                    layer_past_k, layer_past_v = layer_past
            x, layer_attn, present_k, present_v = layer(
                x, key_padding_mask=key_padding_mask, past_k=layer_past_k, past_v=layer_past_v,
                use_cache=use_cache, output_attentions=output_attentions
            )
            if output_attentions:
                all_attentions.append(layer_attn)
            if use_cache:
                present_key_values.append((present_k, present_v))
        x = self.final_ln(x)
        logits = self.output_linear(x)
        if use_cache:
            return logits, all_attentions, present_key_values
        else:
            return logits, all_attentions

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    tokenized_ds_path = "tokenized_hf_dataset"
    VOCAB_SAVE_PATH = "/kaggle/input/300dim-utils/vocab_300dim.pkl"
    EMBEDDING_MATRIX_SAVE_PATH = "/kaggle/input/300dim-utils/embedding_matrix_300dim.pkl"
    print(f"Found saved vocabulary file at '{VOCAB_SAVE_PATH}'. Loading...")
    with open(VOCAB_SAVE_PATH, "rb") as f:
        vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    VOCAB_SIZE = len(word2idx)
    print(f"Found saved embedding matrix file at '{EMBEDDING_MATRIX_SAVE_PATH}'. Loading...")
    with open(EMBEDDING_MATRIX_SAVE_PATH, "rb") as f:
        embedding_matrix = pickle.load(f)
    print("Loading tokenized dataset...")
    from datasets import load_from_disk
    train_ds = load_from_disk(os.path.join(tokenized_ds_path, "train"))
    val_ds = load_from_disk(os.path.join(tokenized_ds_path, "validation"))
    print("Building Dataset")
    train_dataset = ChunkedSequenceDataset(train_ds, word2idx, CONTEXT_LEN, SOS_ID, EOS_ID, UNK_ID)
    val_dataset = ChunkedSequenceDataset(val_ds, word2idx, CONTEXT_LEN, SOS_ID, EOS_ID, UNK_ID)
    collate_fn = partial(collate_batch, pad_id=PAD_ID, context_len=CONTEXT_LEN)
    print("Building Sampler")
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    print("Building Loader")
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    USE_WANDB = True
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    os.environ["WANDB_MODE"] = "offline"
    if USE_WANDB:
        wandb.require("service")
        wandb.init(
            project="transformer-training",
            mode="offline",
            config={
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS
            }
        )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=4, pin_memory=True)
    print("Starting Model")
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_len=CONTEXT_LEN,
        dropout=DROPOUT,
        embedding_weights=embedding_matrix,
        freeze_embeddings=True
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    pad_idx = PAD_ID
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    train_losses, val_losses, perplexities = [], [], []
    print("Training Begins....")
    CKPT_PATH = "/kaggle/input/checkpoint/epoch_8.pt"
    START_EPOCH = 1
    if os.path.exists(CKPT_PATH):
        print(f"Loading checkpoint from {CKPT_PATH} ...")
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        perplexities = checkpoint.get("perplexities", [])
        START_EPOCH = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} — continuing from epoch {START_EPOCH}")
    else:
        print("No checkpoint found — starting training from scratch")
    for epoch in range(START_EPOCH, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(DEVICE)
        model.train()
        running_loss = 0.0
        total_steps = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} train")
        for step, (input_ids, target_ids, _) in pbar:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            target_ids = target_ids.to(DEVICE, non_blocking=True)
            attention_mask = (input_ids != pad_idx).long()
            with torch.cuda.amp.autocast():
                logits,_ = model(input_ids, attention_mask=attention_mask)
                logits_flat = logits.view(-1, VOCAB_SIZE)
                targets_flat = target_ids.view(-1)
                loss = criterion(logits_flat, targets_flat)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()
            total_steps += 1
            avg_loss = running_loss / total_steps
            pbar.set_postfix({'train_loss': avg_loss})
        train_loss = running_loss / total_steps
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for input_ids, target_ids, _ in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                input_ids = input_ids.to(DEVICE, non_blocking=True)
                target_ids = target_ids.to(DEVICE, non_blocking=True)
                attention_mask = (input_ids != pad_idx).long()
                with torch.cuda.amp.autocast():
                    logits,all_attns = model(input_ids, attention_mask=attention_mask, output_attentions = True)
                    logits_flat = logits.view(-1, VOCAB_SIZE)
                    targets_flat = target_ids.view(-1)
                    loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
                val_steps += 1
        val_loss /= val_steps
        val_losses.append(val_loss)
        perplexity = math.exp(val_loss)
        perplexities.append(perplexity)
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, PPL={perplexity:.2f}, PeakMem={peak_mem:.2f} GB")
        if USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "peak_gpu_memory_gb": peak_mem
            })
        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "perplexities": perplexities
        }, ckpt_path)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss", marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), perplexities, label="Perplexity", color="purple", marker='o')
    plt.title("Validation Perplexity Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "perplexity_curve.png"))
    plt.show()
    if USE_WANDB:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()



'''

@torch.no_grad()
def generate(model, prompt_text, tokenizer, word2idx, idx2word,
             max_new_tokens=50, temperature=1.0, top_k=50, eos_token="[EOS]",
             max_total_len=64, use_kv=False):

    model.eval()
    device = next(model.parameters()).device

    tokens = [t.text for t in tokenizer(prompt_text) if not t.is_space]
    prompt_ids = [word2idx.get(tok, word2idx.get("[UNK]", 0)) for tok in tokens]
    if len(prompt_ids) == 0:
        prompt_ids = [word2idx.get("[SOS]", 0)] if "[SOS]" in word2idx else [0]

    all_ids = prompt_ids.copy()
    generated_ids = []

    vocab_size = len(idx2word)
    pad_id = word2idx.get("[PAD]", 0)

    if not use_kv:
        for step in range(max_new_tokens):
            if len(all_ids) >= max_total_len:
                break
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
            attention_mask = (input_ids != pad_id).long()
            logits, _ = model(input_ids, attention_mask=attention_mask)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / float(temperature)
            k = min(int(top_k), next_token_logits.size(-1))
            topk_vals, topk_idx = torch.topk(next_token_logits, k=k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            sampled_idx_in_topk = torch.multinomial(probs[0], num_samples=1).item()
            next_token = int(topk_idx[0, sampled_idx_in_topk].item())
            all_ids.append(next_token)
            generated_ids.append(next_token)
            if idx2word.get(next_token, "") == eos_token:
                break
    else:
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()
        out = model(input_ids, attention_mask=attention_mask, use_cache=True)
        if len(out) == 3:
            logits, _, present_key_values = out
        else:
            logits, _, present_key_values = out[0], out[1], out[2]
        next_token_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_token_logits = next_token_logits / float(temperature)
        k = min(int(top_k), next_token_logits.size(-1))
        topk_vals, topk_idx = torch.topk(next_token_logits, k=k, dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        sampled_idx_in_topk = torch.multinomial(probs[0], num_samples=1).item()
        next_token = int(topk_idx[0, sampled_idx_in_topk].item())
        all_ids.append(next_token)
        generated_ids.append(next_token)

        for step in range(1, max_new_tokens):
            if len(all_ids) >= max_total_len:
                break
            last_id = torch.tensor([[all_ids[-1]]], dtype=torch.long, device=device)
            out = model(last_id, past_key_values=present_key_values, use_cache=True)
            if len(out) == 3:
                logits, _, present_key_values = out
            else:
                logits, _, present_key_values = out[0], out[1], out[2]
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / float(temperature)
            k = min(int(top_k), next_token_logits.size(-1))
            topk_vals, topk_idx = torch.topk(next_token_logits, k=k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            sampled_idx_in_topk = torch.multinomial(probs[0], num_samples=1).item()
            next_token = int(topk_idx[0, sampled_idx_in_topk].item())
            all_ids.append(next_token)
            generated_ids.append(next_token)
            if idx2word.get(next_token, "") == eos_token:
                break

    gen_tokens = [idx2word.get(i, "[UNK]") for i in generated_ids]
    return gen_tokens


'''