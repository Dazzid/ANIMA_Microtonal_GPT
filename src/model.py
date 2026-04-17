"""
model.py
========
GPT-2 model for 53-TET microtonal music generation.

Dual-channel architecture:
  Channel 1 — Token IDs    → token embedding → self-attention
  Channel 2 — EigenSpace 4D → harmonic positional encoding (α, β, γ, δ)

Positional encoding design:
  - EigenSpace: harmonic position in psychoacoustic space (chord-level)
  - Local position: intra-chord token ordering (CHORD_START=0, DUR=1, ROOT=2, PV=3...)
  - Sequential position: song-level ordering (standard learned positional embedding)

Model sizes:
  ModelConfig()            → ~85M params  (12L/12H/768E — GPT-2 small)
  ModelConfig.medium()     → ~307M params (24L/16H/1024E — GPT-2 medium)
  ModelConfig.large()      → ~774M params (36L/20H/1280E — GPT-2 large)
  ModelConfig.small()      → ~13M params  (6L/6H/384E — debugging)
"""

import math
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """GPT-2 model configuration for 53-TET music."""
    # Core architecture
    vocab_size: int = 2711
    block_size: int = 4096
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False
    # EigenSpace positional encoding
    use_eigenspace: bool = True
    n_eigen: int = 4              # (α, β, γ, δ)
    eigen_hidden: int = 128       # hidden dim in eigenspace projection MLP
    # Intra-chord positional encoding
    max_local_pos: int = 20       # max tokens within a chord (CHORD_START..CHORD_END)
    # Sequential positional encoding
    use_sequential_pos: bool = True

    @classmethod
    def medium(cls):
        """~307M params — GPT-2 medium."""
        return cls(n_layer=24, n_head=16, n_embd=1024, eigen_hidden=192)

    @classmethod
    def large(cls):
        """~774M params — GPT-2 large."""
        return cls(n_layer=36, n_head=20, n_embd=1280, eigen_hidden=256)

    @classmethod
    def small(cls):
        """~13M params — fast iteration / debugging."""
        return cls(n_layer=6, n_head=6, n_embd=384, eigen_hidden=64)


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class EigenSpacePositionalEncoding(nn.Module):
    """
    Projects 4D harmonic coordinates (α, β, γ, δ) into positional encoding.

    EigenSpace coordinates define where a chord sits in harmonic space —
    the transformer's sense of "position" for chord-level structure
    and harmonic progression.

    Architecture: 4 → hidden → hidden → n_embd
    """

    def __init__(self, n_embd: int, n_eigen: int = 4, hidden: int = 128):
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
    """Multi-head causal self-attention with Flash Attention support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
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

    Embedding:
      h = tok_emb(tokens) + eigen_pos(α,β,γ,δ) + local_pos(chord_offset) [+ seq_pos]
      h = transformer_blocks(h)
      logits = lm_head(layer_norm(h))
    """

    CHORD_START_ID = 4
    CHORD_END_ID = 5

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            local_pos = nn.Embedding(config.max_local_pos, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # EigenSpace positional encoding
        self.use_eigenspace = config.use_eigenspace
        if self.use_eigenspace:
            self.eigen_pos = EigenSpacePositionalEncoding(
                config.n_embd,
                n_eigen=config.n_eigen,
                hidden=config.eigen_hidden,
            )

        # Sequential positional encoding
        self.use_sequential_pos = getattr(config, 'use_sequential_pos', False)
        if self.use_sequential_pos:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Language model head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

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
        incrementing positions: 0, 1, 2, ...
        Non-chord tokens get position 0.

        Returns: (B, T) local position indices, clamped to [0, max_local_pos-1]
        """
        B, T = idx.shape
        device = idx.device
        max_pos = self.config.max_local_pos - 1

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
        tok_emb = self.transformer.wte(idx)

        # Harmonic positional encoding (EigenSpace)
        harmonic_pos = 0.0
        if self.use_eigenspace and eigen is not None:
            harmonic_pos = self.eigen_pos(eigen)

        # Intra-chord local position encoding
        local_ids = self._compute_local_positions(idx)
        local_emb = self.transformer.local_pos(local_ids)

        # Combine embeddings
        x = tok_emb + harmonic_pos + local_emb

        # Sequential position
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
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # ignore <pad> token (id=0)
            )
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters into decay / no-decay groups.
        Decay on matmul weights, no decay on biases + norms + embeddings.
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

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
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
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            eigen_cond = None
            if eigen is not None:
                eigen_cond = eigen if eigen.size(1) <= self.config.block_size else eigen[:, -self.config.block_size:]

            logits, _ = self(idx_cond, eigen=eigen_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eigen is not None:
                default = torch.tensor([1.0, 1.0, 2.0, 1.0], device=eigen.device)
                default = default.view(1, 1, 4).expand(idx.size(0), 1, 4)
                eigen = torch.cat((eigen, default), dim=1)

        return idx
