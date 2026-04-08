"""
train.py
========
GPT-2 training script for 53-TET microtonal music generation.

Dual-channel architecture:
  Channel 1 — Token IDs    → token embedding → self-attention (sequential patterns)
  Channel 2 — EigenSpace 4D → spatial embedding (harmonic geometry, α β γ δ)

The EigenSpace coordinates encode the psychoacoustic DNA of each chord
(3rd quality, 5th quality, 7th quality, root pitch class) and are injected
as an additive embedding alongside the token + positional embeddings.

Data
----
Reads memory-mapped binary files produced by pack_data.py:
  dataset/tokenized/
    train_tokens.bin, train_eigen.bin
    val_tokens.bin,   val_eigen.bin
    meta.json,        vocab.json

Usage
-----
  # Default training:
  python train.py

  # Override hyperparameters:
  python train.py --n-layer 8 --n-head 8 --n-embd 512 --batch-size 64

  # Resume from checkpoint:
  python train.py --resume checkpoints/latest.pt

  # Quick test:
  python train.py --max-iters 100 --eval-interval 10
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Path setup ──
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """GPT-2 model configuration for 53-TET music."""
    vocab_size: int = 345
    block_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    # EigenSpace as positional encoding (v2 architecture)
    use_eigenspace: bool = True
    n_eigen: int = 4              # (α, β, γ, δ)
    eigen_hidden: int = 64        # hidden dim in eigenspace projection MLP
    # Intra-chord positional encoding
    max_local_pos: int = 20       # max tokens within a chord (CHORD_START..CHORD_END)
    # Sequential positional encoding (song-level position)
    use_sequential_pos: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = str(_ROOT_DIR / "dataset" / "tokenized")
    # Optimization
    max_iters: int = 100_000
    batch_size: int = 16
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # LR schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 100_000
    min_lr: float = 3e-5
    # Eval
    eval_interval: int = 500
    eval_iters: int = 200
    # Logging
    log_interval: int = 10
    wandb_project: str = "anima-gpt2-53tet"
    wandb_run_name: str = ""
    use_wandb: bool = True
    # Checkpoints
    checkpoint_dir: str = str(_ROOT_DIR / "checkpoints")
    save_interval: int = 5000
    resume: str = ""
    # System
    device: str = "auto"
    compile_model: bool = False    # torch.compile (PyTorch 2.0+)
    num_workers: int = 0


# =============================================================================
# MEMORY-MAPPED DATASET
# =============================================================================

class DualChannelBinaryDataset:
    """
    Memory-mapped dataset for dual-channel GPT-2 training.

    Reads fixed-length padded sequences created by pack_data.py.
    Each song is stored as one or more sequences of exactly (block_size + 1)
    tokens. Short songs are padded with <pad> (id=0). No window ever crosses
    a song boundary.

    The data is stored as a 2D array: (n_sequences, block_size + 1).
    get_batch() picks random sequence indices — not random byte offsets.

    Returns:
        x_tokens:  (block_size,) int64   — input token IDs
        x_eigen:   (block_size, 4) float32 — input eigenspace coords
        y_tokens:  (block_size,) int64   — target token IDs (shifted by 1)
    """

    def __init__(self, tokens_path: str, eigen_path: str, block_size: int,
                 n_seqs: int = None):
        self.block_size = block_size
        seq_len = block_size + 1

        # Memory-map the token file (uint16) and reshape to (n_seqs, seq_len)
        tokens_flat = np.memmap(tokens_path, dtype=np.uint16, mode='r')
        if n_seqs is None:
            n_seqs = len(tokens_flat) // seq_len
        self.tokens = tokens_flat[:n_seqs * seq_len].reshape(n_seqs, seq_len)

        # Memory-map the eigenspace file (float16, 4 dims per token)
        eigen_flat = np.memmap(eigen_path, dtype=np.float16, mode='r')
        self.eigen = eigen_flat[:n_seqs * seq_len * 4].reshape(n_seqs, seq_len, 4)

        self.n_seqs = n_seqs

        n_real_tokens = int((self.tokens != 0).sum())
        print(f"  Loaded {n_seqs:,} sequences × {seq_len} "
              f"({n_real_tokens:,} real tokens, "
              f"{self.tokens.size - n_real_tokens:,} padding)")

    def __len__(self):
        return self.n_seqs

    def get_batch(self, batch_size: int, device: torch.device):
        """
        Get a random batch of complete sequences (no boundary crossing).

        Returns:
            x: (B, block_size) int64
            x_eigen: (B, block_size, 4) float32
            y: (B, block_size) int64
        """
        ix = np.random.randint(0, self.n_seqs, size=(batch_size,))

        # Vectorised bulk read from memmap – one fancy-index per array
        tok_batch = np.array(self.tokens[ix])          # (B, seq_len) uint16
        eig_batch = np.array(self.eigen[ix, :self.block_size])  # (B, block_size, 4) fp16

        x     = torch.from_numpy(tok_batch[:, :self.block_size].astype(np.int64)).to(device)
        y     = torch.from_numpy(tok_batch[:, 1:self.block_size + 1].astype(np.int64)).to(device)
        eigen = torch.from_numpy(eig_batch.astype(np.float32)).to(device)

        return x, eigen, y


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class EigenSpacePositionalEncoding(nn.Module):
    """
    Projects 4D harmonic coordinates (α, β, γ, δ) into positional encoding.

    This IS the model's primary positional encoding. EigenSpace coordinates
    define where a chord sits in harmonic space — the transformer's sense
    of "position" is harmonic position, not sequential index.

    The 2-hidden-layer MLP learns non-linear mappings from the continuous
    psychoacoustic coordinates to attention-compatible representations.

    Architecture: 4 → hidden → hidden → n_embd
    """

    def __init__(self, n_embd: int, n_eigen: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_eigen, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 4) → (B, T, n_embd)"""
        return self.net(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V projections combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Use Flash Attention if available (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # Manual causal mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: LN → Attn → LN → MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# GPT-2 MODEL
# =============================================================================

class GPT2(nn.Module):
    """
    GPT-2 for 53-TET microtonal music generation.

    Harmonic Architecture (v2):
      h = tok_emb(tokens) + eigen_pos(α,β,γ) + local_pos(chord_offset) + drop
      h = transformer_blocks(h)
      logits = lm_head(layer_norm(h))

    Positional encoding design:
      - EigenSpace IS the positional encoding. Harmonic coordinates (α, β, γ)
        define where the model is in psychoacoustic space — this is its sense of
        "position" for chord-level structure and harmonic progression.
      - Intra-chord local positions provide token ordering WITHIN each chord
        (CHORD_START=0, DUR=1, ROOT=2, PV_a=3, PV_b=4, ..., CHORD_END=n).
      - Sequential position (wpe) is REMOVED by default. Causal masking provides
        token ordering; harmonic position provides musical structure.
    """

    # Token IDs for chord boundary detection (from 53-TET vocabulary)
    CHORD_START_ID = 4
    CHORD_END_ID = 5

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),          # token embedding
            local_pos = nn.Embedding(config.max_local_pos, config.n_embd), # intra-chord position
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # EigenSpace positional encoding — the harmonic positional signal
        self.use_eigenspace = config.use_eigenspace
        if self.use_eigenspace:
            self.eigen_pos = EigenSpacePositionalEncoding(
                config.n_embd,
                n_eigen=config.n_eigen,
                hidden=config.eigen_hidden,
            )

        # Optional: legacy sequential positional encoding (disabled by default)
        self.use_sequential_pos = getattr(config, 'use_sequential_pos', False)
        if self.use_sequential_pos:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Language model head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Init weights
        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_local_positions(self, idx):
        """
        Compute intra-chord token positions from token IDs.

        Within each chord (CHORD_START to CHORD_END), tokens get
        incrementing positions: 0 for CHORD_START, 1 for DUR,
        2 for first pitch, 3 for first velocity, etc.

        Non-chord tokens (BAR, <start>, <end>, etc.) get position 0.

        Args:
            idx: (B, T) token IDs

        Returns:
            (B, T) local position indices, clamped to [0, max_local_pos-1]
        """
        B, T = idx.shape
        device = idx.device
        max_pos = self.config.max_local_pos - 1

        # Move to CPU/numpy for fast Python scan (O(B*T), ~5ms for B=64,T=1024)
        idx_np = idx.detach().cpu().numpy()
        pos_np = np.zeros((B, T), dtype=np.int64)

        cs_id = self.CHORD_START_ID
        ce_id = self.CHORD_END_ID

        for b in range(B):
            pos = 0
            in_chord = False
            for t in range(T):
                tok = idx_np[b, t]
                if tok == cs_id:
                    in_chord = True
                    pos = 0
                elif tok == ce_id:
                    pos += 1
                    in_chord = False
                elif in_chord:
                    pos += 1
                else:
                    pos = 0
                pos_np[b, t] = min(pos, max_pos)

        return torch.from_numpy(pos_np).to(device)

    def forward(self, idx, eigen=None, targets=None):
        """
        Args:
            idx:     (B, T) int64 — token IDs
            eigen:   (B, T, 4) float32 — eigenspace coordinates (α, β, γ, δ)
            targets: (B, T) int64 — target token IDs (optional, for loss)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar or None
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token embedding
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)

        # Harmonic positional encoding (EigenSpace)
        harmonic_pos = 0.0
        if self.use_eigenspace and eigen is not None:
            harmonic_pos = self.eigen_pos(eigen)  # (B, T, n_embd)

        # Intra-chord local position encoding
        local_ids = self._compute_local_positions(idx)  # (B, T)
        local_emb = self.transformer.local_pos(local_ids)  # (B, T, n_embd)

        # Combine: token identity + harmonic position + local ordering
        x = tok_emb + harmonic_pos + local_emb

        # Optional: add legacy sequential position
        if self.use_sequential_pos:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.wpe(pos)

        x = self.transformer.drop(x)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Compute loss if targets given
        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # ignore <pad> token (id=0)
            )
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters into decay / no-decay groups.
        Follows GPT-2/3 convention: decay on matmul weights, no decay on biases + norms + embeddings.
        """
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist):
                    no_decay.add(fpn)

        # Remove weight-tied parameters (lm_head shares wte weights)
        decay.discard('lm_head.weight')
        no_decay.discard('lm_head.weight')

        # Validate
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        union = decay | no_decay
        # Only check params that are not tied
        untied_params = {pn for pn, p in self.named_parameters()
                         if p.data_ptr() not in
                         [param_dict[other].data_ptr() for other in union if other in param_dict]}
        assert len(inter) == 0, f"Parameters in both sets: {inter}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay) if pn in param_dict],
             "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay) if pn in param_dict],
             "weight_decay": 0.0},
        ]

        use_fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using {'fused' if use_fused else 'standard'} AdamW optimizer")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, eigen=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Autoregressive generation.

        Args:
            idx:   (B, T) conditioning token IDs
            eigen: (B, T, 4) conditioning eigenspace (optional)
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (None = no filtering)

        Returns:
            (B, T + max_new_tokens) generated token IDs
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            eigen_cond = None
            if eigen is not None:
                eigen_cond = eigen if eigen.size(1) <= self.config.block_size else eigen[:, -self.config.block_size:]

            # Forward
            logits, _ = self(idx_cond, eigen=eigen_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Extend eigen with default values for generated tokens
            if eigen is not None:
                default = torch.tensor([1.0, 1.0, 2.0, 1.0], device=eigen.device)
                default = default.view(1, 1, 4).expand(idx.size(0), 1, 4)
                eigen = torch.cat((eigen, default), dim=1)

        return idx


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(it: int, config: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # After decay: min LR
    if it > config.lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config: TrainConfig, device):
    """Estimate train and val loss over eval_iters batches."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, eigen, y = data.get_batch(config.batch_size, device)
            logits, loss = model(x, eigen=eigen, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on 53-TET dual-channel data")

    # Model
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-eigenspace", action="store_true", help="Disable EigenSpace embedding")

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--min-lr", type=float, default=3e-5)

    # Eval & logging
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)

    # Data
    parser.add_argument("--data-dir", type=str, default=str(_ROOT_DIR / "dataset" / "tokenized"))

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default=str(_ROOT_DIR / "checkpoints"))
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    # Logging
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable W&B logging (default: on)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="anima-gpt2-53tet")
    parser.add_argument("--wandb-run-name", type=str, default="")

    # System
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    print(f"Device: {device} ({device_type})")

    if device_type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Data ──
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.json"
    assert meta_path.exists(), f"meta.json not found in {data_dir}. Run pack_data.py first."

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    vocab_size = meta['vocab_size']
    block_size = args.block_size

    print(f"\n{'='*72}")
    print("ANIMA GPT-2 — 53-TET Microtonal Music")
    print(f"{'='*72}")
    print(f"  Vocab size:    {vocab_size}")
    print(f"  Block size:    {block_size}")
    print(f"  Train tokens:  {meta['n_train_tokens']:,}")
    print(f"  Val tokens:    {meta['n_val_tokens']:,}")

    print(f"\nLoading datasets...")
    train_data = DualChannelBinaryDataset(
        str(data_dir / "train_tokens.bin"),
        str(data_dir / "train_eigen.bin"),
        block_size,
        n_seqs=meta.get('n_train_seqs'),
    )
    val_data = DualChannelBinaryDataset(
        str(data_dir / "val_tokens.bin"),
        str(data_dir / "val_eigen.bin"),
        block_size,
        n_seqs=meta.get('n_val_seqs'),
    )

    # ── Model ──
    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_eigenspace=not args.no_eigenspace,
    )

    print(f"\nModel config:")
    print(f"  Layers:      {model_config.n_layer}")
    print(f"  Heads:       {model_config.n_head}")
    print(f"  Embedding:   {model_config.n_embd}")
    print(f"  EigenSpace:  {model_config.use_eigenspace}")
    print(f"  Dropout:     {model_config.dropout}")

    model = GPT2(model_config)
    model.to(device)

    # Optional: torch.compile
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Training config ──
    train_config = TrainConfig(
        data_dir=args.data_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_iters=args.warmup_iters,
        min_lr=args.min_lr,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb and not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compile_model=args.compile,
    )

    tokens_per_iter = train_config.batch_size * train_config.gradient_accumulation_steps * block_size
    print(f"\n  Batch size:    {train_config.batch_size}")
    print(f"  Grad accum:    {train_config.gradient_accumulation_steps}")
    print(f"  Tokens/iter:   {tokens_per_iter:,}")
    print(f"  Max iters:     {train_config.max_iters:,}")
    print(f"  Total tokens:  ~{tokens_per_iter * train_config.max_iters / 1e9:.2f}B")

    # ── Optimizer ──
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (train_config.beta1, train_config.beta2),
        device_type,
    )

    # ── Resume ──
    iter_num = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resumed at iter {iter_num}, best_val_loss={best_val_loss:.4f}")

    # ── Wandb ──
    if train_config.use_wandb:
        import wandb
        run_name = train_config.wandb_run_name or \
            f"L{model_config.n_layer}_H{model_config.n_head}_E{model_config.n_embd}" \
            f"{'_eigen' if model_config.use_eigenspace else ''}"
        wandb.init(
            project=train_config.wandb_project,
            name=run_name,
            config={
                **{k: v for k, v in model_config.__dict__.items()},
                **{k: v for k, v in train_config.__dict__.items()},
                "n_params": sum(p.numel() for p in model.parameters()),
            },
        )

    # ── Checkpoint directory ──
    ckpt_dir = Path(train_config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Mixed precision context ──
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16) \
        if device_type == 'cuda' else torch.nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda'))

    # ── Training loop ──
    print(f"\n{'='*72}")
    print("Starting training...")
    print(f"{'='*72}\n")

    model.train()
    t0 = time.time()
    local_iter = 0

    while iter_num < train_config.max_iters:

        # ── Learning rate schedule ──
        lr = get_lr(iter_num, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ── Evaluation ──
        if iter_num % train_config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_data, val_data, train_config, device)
            print(f"\n  [eval] iter {iter_num:>6,}: "
                  f"train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

            if train_config.use_wandb:
                wandb.log({
                    "eval/train_loss": losses['train'],
                    "eval/val_loss": losses['val'],
                    "lr": lr,
                }, step=iter_num)

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_checkpoint(model, optimizer, iter_num, best_val_loss,
                                model_config, ckpt_dir / "best.pt")
                print(f"  [save] New best val_loss={best_val_loss:.4f}")

        # ── Save periodic checkpoint ──
        if iter_num % train_config.save_interval == 0 and iter_num > 0:
            save_checkpoint(model, optimizer, iter_num, best_val_loss,
                            model_config, ckpt_dir / "latest.pt")

        # ── Gradient accumulation loop ──
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            x, eigen, y = train_data.get_batch(train_config.batch_size, device)

            with ctx:
                logits, loss = model(x, eigen=eigen, targets=y)
                loss = loss / train_config.gradient_accumulation_steps

            loss_accum += loss.item()
            scaler.scale(loss).backward()

        # ── Gradient clipping ──
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──
        if iter_num % train_config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_sec = tokens_per_iter * train_config.log_interval / dt if dt > 0 and local_iter > 0 else 0
            print(f"  iter {iter_num:>6,} | loss {loss_accum:.4f} | "
                  f"lr {lr:.2e} | {tokens_sec:,.0f} tok/s")

            if train_config.use_wandb:
                wandb.log({
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_sec,
                }, step=iter_num)

        iter_num += 1
        local_iter += 1

    # ── Final save ──
    save_checkpoint(model, optimizer, iter_num, best_val_loss,
                    model_config, ckpt_dir / "final.pt")
    print(f"\n{'='*72}")
    print(f"Training complete at iter {iter_num:,}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints:   {ckpt_dir}/")
    print(f"{'='*72}")

    if train_config.use_wandb:
        wandb.finish()


def save_checkpoint(model, optimizer, iter_num, best_val_loss, model_config, path):
    """Save a training checkpoint."""
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'model_config': model_config.__dict__,
    }, str(path))


if __name__ == "__main__":
    main()
