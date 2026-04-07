# ANIMA GPT-2 — Harmonic Architecture v2

## What Changed

The model's positional encoding was fundamentally redesigned. EigenSpace is now 
the **primary positional signal**, not a secondary additive embedding.

### Before (v1 — broken)
```
h = tok_emb(tokens) + wpe(0,1,2,...,T) + eigen_emb(α,β,γ,D)
                      ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                      PRIMARY position   tiny additive noise
                      (sequential)       (4→16→384, 6K params)
```
The model's sense of "position" was sequential index. EigenSpace was a weak 
additive signal that attention couldn't meaningfully use. Result: harmonically 
static parallel chord blocks.

### After (v2 — harmonic architecture)
```
h = tok_emb(tokens) + eigen_pos(α,β,γ,D) + local_pos(chord_offset)
                      ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
                      PRIMARY position       intra-chord ordering
                      (harmonic space)       (0=CHORD_START, 1=DUR,
                      (4→64→64→384, 29K)     2=P_a, 3=V_a, ...)
```
The model's sense of "position" is now **harmonic position** — where it sits 
in the psychoacoustic EigenSpace tetrahedron. Sequential ordering comes from 
causal masking (can only attend to past tokens).

## Architecture Details

### EigenSpace Positional Encoding
- **Input**: (α, β, γ, D) — 3rd ratio, 5th ratio, 7th ratio, Plomp-Levelt dissonance
- **MLP**: 4 → 64 → 64 → 384 (2 hidden layers, GELU activations)
- **Role**: Defines WHERE in harmonic space the model is. A Cm7 chord maps to 
  a different position than a Cmaj7, and the model learns to generate progressions 
  that *move through* this space coherently
- **Parameters**: ~29K (5x the old 6K, but still <0.3% of total model)

### Intra-chord Local Position
- **Input**: position index within chord (0 for CHORD_START, 1 for DUR, 2 for 
  first P_, 3 for first V_, etc.)
- **Embedding**: nn.Embedding(20, 384) — 20 possible positions, plenty for 
  8-note chords
- **Role**: Within a chord (all tokens share same eigenspace), distinguishes 
  "I'm the 3rd voice" from "I'm the 5th voice" — critical for voice leading
- **Non-chord tokens**: All get position 0 (they have different eigenspace 
  defaults and different token embeddings, so no confusion)
- **Parameters**: 7,680

### Sequential Position (wpe)
- **REMOVED** by default (`use_sequential_pos=False`)
- Can be re-enabled for comparison experiments (`use_sequential_pos=True`)
- Causal attention mask already provides "I can only see the past" ordering

## Model Parameters

| Component | v1 | v2 |
|-----------|----|----|
| Token embedding (wte) | 132,480 | 132,480 |
| Sequential position (wpe) | 393,216 | 0 (removed) |
| EigenSpace projection | 6,528 | 29,056 |
| Local position (new) | 0 | 7,680 |
| Attention + MLP blocks | ~10.6M | ~10.6M |
| **Total** | **11.2M** | **10.8M** |

## Files Modified

- `src/09_train_gpt2.py` — Model architecture, training loop (unchanged)
- `src/10_generate.py` — Checkpoint loading config 
- `src/eigenspace.py` — Updated EigenSpaceEmbedding to match v2 MLP

## Data Format

**No changes to training data** — the existing `train_tokens.bin` and 
`train_eigen.bin` files work as-is. Local positions are computed on-the-fly 
from token IDs in the model's forward pass.

## To Retrain

```bash
cd /home/david/Projects/ANIMA_Data_Formation
python src/09_train_gpt2.py --wandb-run-name "v2_harmonic_pos"
```

Old checkpoints are **incompatible** — the state dict keys changed (no more 
`transformer.wpe`, new `eigen_pos.*` and `transformer.local_pos.*`). This is 
expected; v1 checkpoints produced garbage anyway.

## Why This Should Work

1. **Harmonic progression**: The model now sees that chord A is at position 
   (1.19, 1.50, 1.78, -1.5) and chord B is at (1.26, 1.50, 1.90, -2.1). It 
   learns that certain _movements_ through this space are musically coherent — 
   ii→V→I is a specific trajectory through eigenspace.

2. **Voice leading**: Intra-chord positions tell the model "I'm generating the 
   3rd voice of this chord" — it can learn that the 3rd voice in chord B should 
   resolve smoothly from where the 3rd voice was in chord A.

3. **No more static blocks**: Without sequential position anchoring tokens to 
   absolute indices, the model can't just memorize "at position 15-25, output 
   this chord pattern." It must respond to the harmonic context.
