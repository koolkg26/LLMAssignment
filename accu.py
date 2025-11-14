%%writefile train_ddp_v6.py
import os
import math
import pickle
import time
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
ACCUM_STEPS = 1

class ChunkedSequenceDataset(Dataset):
    def __init__(self, hf_dataset, word2idx, context_len, sos_id, eos_id, unk_id):
        self.word2idx = word2idx
        self.context_len = context_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        print(f"Preparing sequences (context_len={context_len})...")
        self.chunks = self._prepare_chunks(hf_dataset)
        print(f"Created {len(self.chunks):,} chunks total.")

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

    def forward(self, x, padding_mask=None, output_attentions: bool = False):
        b, t, _ = x.size()
        q = self.wq(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        orig_dtype = scores.dtype
        scores = scores.float()
        causal = torch.tril(torch.ones((t, t), dtype=torch.bool, device=x.device))
        allowed = causal.unsqueeze(0).unsqueeze(0)
        if padding_mask is not None:
            pad_bool = padding_mask if padding_mask.dtype == torch.bool else (padding_mask == 0)
            key_is_real = (~pad_bool).unsqueeze(1).unsqueeze(2)
            allowed = allowed & key_is_real
            allowed = allowed.expand(b, self.num_heads, t, t)
        else:
            allowed = allowed.expand(b, self.num_heads, t, t)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn.to(v.dtype)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.wo(context)
        return out, attn

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

    def forward(self, x, key_padding_mask=None,output_attentions=False):
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.self_attn(x_norm, padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x_norm2 = self.ln2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x,attn_weights

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

    def forward(self, input_ids, attention_mask=None,output_attentions=False):
        b, t = input_ids.size()
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_encoding[:t, :].unsqueeze(0).expand(b, -1, -1)
        x = tok_emb + pos_emb
        if attention_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = (attention_mask == 0)
        all_attentions = []
        for layer in self.layers:
            if output_attentions:
                x, layer_attn = layer(x, key_padding_mask=key_padding_mask, output_attentions=True)
                all_attentions.append(layer_attn)
            else:
                x,_ = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)
        logits = self.output_linear(x)
        return logits, all_attentions

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    tokenized_ds_path = "tokenized_hf_dataset"
    VOCAB_SAVE_PATH = "/kaggle/input/300dim-utils/vocab_300dim.pkl"
    EMBEDDING_MATRIX_SAVE_PATH = "/kaggle/input/300dim-utils/embedding_matrix_300dim.pkl"
    print(f"\nFound saved vocabulary file at '{VOCAB_SAVE_PATH}'. Loading...")
    with open(VOCAB_SAVE_PATH, "rb") as f:
        vocab_data = pickle.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
    print(f"Loaded vocabulary with {len(word2idx):,} tokens.")
    VOCAB_SIZE = len(word2idx)
    EMBEDDING_DIM = 100
    print(f"\nFound saved embedding matrix file at '{EMBEDDING_MATRIX_SAVE_PATH}'. Loading...")
    with open(EMBEDDING_MATRIX_SAVE_PATH, "rb") as f:
        embedding_matrix = pickle.load(f)
    print(f"Loaded embedding matrix with shape {embedding_matrix.shape}.")
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
    USE_WANDB = False
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    os.environ["WANDB_MODE"] = "offline"
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
    train_losses, epoch_durations = [], []
    print("Training Begins....")
    CKPT_PATH = "/kaggle/input/checkpoint/epoch_8.pt"
    START_EPOCH = 1
    if rank == 0: print("Training Begins....")
    for epoch in range(START_EPOCH, EPOCHS + 1):
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        total_steps = 0
        epoch_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_loader), 
                    total=len(train_loader), 
                    desc=f"Epoch {epoch} train", 
                    disable=(rank != 0))
        for step, (input_ids, target_ids, _) in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            attention_mask = (input_ids != pad_idx).long()
            with torch.cuda.amp.autocast():
                logits,_ = model(input_ids, attention_mask=attention_mask)
                logits_flat = logits.view(-1, VOCAB_SIZE)
                targets_flat = target_ids.view(-1)
                loss = criterion(logits_flat, targets_flat)
                loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()
            is_last_step_in_batch = (step + 1) % ACCUM_STEPS == 0
            is_last_step_of_epoch = (step + 1) == len(train_loader)
            if is_last_step_in_batch or is_last_step_of_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += (loss.item() * ACCUM_STEPS)
            total_steps += 1
            if rank == 0:
                avg_loss = running_loss / total_steps
                pbar.set_postfix({'train_loss': avg_loss})
        train_loss = running_loss / total_steps
        train_losses.append(train_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        if rank == 0:
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Runtime={epoch_duration:.2f}s, PeakMem={peak_mem:.2f} GB")
    if rank == 0:
        results = {
            "train_losses": train_losses,
            "epoch_durations": epoch_durations,
            "accum_steps": ACCUM_STEPS,
            "physical_batch_size": BATCH_SIZE,
            "effective_batch_size_per_gpu": BATCH_SIZE * ACCUM_STEPS,
            "world_size": world_size,
            "total_effective_batch_size": BATCH_SIZE * ACCUM_STEPS * world_size,
            "epochs": EPOCHS
        }
        save_path = f"results_accum_{ACCUM_STEPS}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved training results to {save_path}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
