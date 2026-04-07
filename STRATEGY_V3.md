# ANIMA v3 — Strategy: Chord-as-Column Architecture

## The Problem

We want to train a GPT-2 that interprets data as MIDI chords. The data is
inherently 2D: a piano roll matrix (time × pitch) with MPE content in each
active cell (velocity, microtonal pitch bend). GPT-2 consumes 1D sequences.

Previous approaches serialized the chord into flat tokens:
`CHORD_START DUR_8.0 P_212 V_6 P_243 V_4 P_265 V_5 CHORD_END`

This is BS because:
1. P_212 is a meaningless label. The number 212 carries zero musical
   information to the embedding layer.
2. Simultaneous notes are serialized into sequential predictions. The model
   predicts P_243 AFTER P_212 as if it's the next event. It's not — it's
   the same event.
3. CHORD_START/CHORD_END are decorative. They don't enforce any structural
   understanding.
4. EigenSpace is computed from the complete chord, but the model builds the
   chord token by token. Chicken and egg.

## The Data (measured, not assumed)

From the 53-TET MPE dataset:
- **Notes per chord**: 4-6 (mean 4.9, 76% have exactly 5)
- **53-TET pitch range**: step 164 to 336 (173 positions, ~3.3 octaves)
- **Velocities**: 55 to 84 (MIDI 0-127 range, narrow in this dataset)
- **Durations**: {2.0, 4.0, 8.0} beats (dominant); some outliers up to 224
- **Chords per song**: ~54
- **Songs in dataset**: ~13,000 (train + val)

## Data Example: "Preciso Me Encontrar" (C minor, major type)

### The Raw MIDI Data (what we have)

```
--- Chord 0 --- onset: 0.0 beats   duration: 8.0 beats
  note 0: step_53=212  (oct=4, pos= 0)  vel= 83  freq= 130.39 Hz  (~C3)
  note 1: step_53=243  (oct=4, pos=31)  vel= 61  freq= 195.57 Hz  (~G3)
  note 2: step_53=265  (oct=5, pos= 0)  vel= 70  freq= 260.77 Hz  (~C4)
  note 3: step_53=283  (oct=5, pos=18)  vel= 71  freq= 329.99 Hz  (~E4)
  note 4: step_53=318  (oct=6, pos= 0)  vel= 59  freq= 521.54 Hz  (~C5)
  intervals from root: [0, 31, 53, 71, 106]
  freq ratios from root: [1.4999  2.0000  2.5309  4.0000]

--- Chord 1 --- onset: 8.0 beats   duration: 8.0 beats
  note 0: step_53=199  (oct=3, pos=40)  vel= 82  freq= 110.00 Hz  (~A2)
  note 1: step_53=252  (oct=4, pos=40)  vel= 77  freq= 220.00 Hz  (~A3)
  note 2: step_53=283  (oct=5, pos=18)  vel= 78  freq= 329.99 Hz  (~E4)
  note 3: step_53=287  (oct=5, pos=22)  vel= 63  freq= 347.71 Hz  (~F4)
  note 4: step_53=318  (oct=6, pos= 0)  vel= 72  freq= 521.54 Hz  (~C5)
  intervals from root: [0, 53, 84, 88, 119]
  freq ratios from root: [2.0000  2.9999  3.1610  4.7413]
```

Each chord is a rich multidimensional event: 4-6 simultaneous notes, each
with a precise 53-TET pitch (octave + position within octave), velocity,
and frequency. The intervals and ratios define the harmonic identity.

### What the current tokenization produces (BS)

```
  [  0] <start>
  [  1] CHORD_START        ← decorative delimiter
  [  2] DUR_8.0
  [  3] P_212              ← arbitrary label, no musical meaning to the model
  [  4] V_6                ← quantized to 8 bins, velocity 83 → bin 6
  [  5] P_243              ← model has no idea this is 31 steps above P_212
  [  6] V_4
  [  7] P_265              ← model has no idea this is an octave above P_212
  [  8] V_5
  [  9] P_283
  [ 10] V_5
  [ 11] P_318              ← model has no idea this is 2 octaves above P_212
  [ 12] V_4
  [ 13] CHORD_END          ← decorative delimiter
  [ 14] BAR
  [ 15] BAR
```

5 simultaneous notes → 12 sequential tokens. The chord structure is
destroyed. The pitch relationships are invisible. The model predicts P_243
after P_212 as if it's the "next event" — it's not, it's the SAME event.

### What the column representation produces (v3)

```
Column 0: duration=8.0 beats                        EigenSpace: (α, β, γ, D)
  173 positions (step 164–336), 5 active:
    [ 48] step=212 (oct=4, pos= 0)  vel=0.654       ← position in array = pitch
    [ 79] step=243 (oct=4, pos=31)  vel=0.480       ← 31 cells away = fifth
    [101] step=265 (oct=5, pos= 0)  vel=0.551       ← 53 cells away = octave
    [119] step=283 (oct=5, pos=18)  vel=0.559
    [154] step=318 (oct=6, pos= 0)  vel=0.465       ← 106 cells = 2 octaves
  Density: 2.9%   (168 zeros, 5 active values)

Column 1: duration=8.0 beats                        EigenSpace: (α, β, γ, D)
  173 positions, 5 active:
    [ 35] step=199 (oct=3, pos=40)  vel=0.646
    [ 88] step=252 (oct=4, pos=40)  vel=0.606
    [119] step=283 (oct=5, pos=18)  vel=0.614
    [123] step=287 (oct=5, pos=22)  vel=0.496
    [154] step=318 (oct=6, pos= 0)  vel=0.567
  Density: 2.9%
```

The chord is ONE vector. Position in the vector = pitch position in the
53-TET grid. Adjacent cells = adjacent pitches. 53 cells apart = octave.
The spatial relationship between notes is preserved in the geometry of the
vector itself. Duration is a separate scalar. EigenSpace maps 1:1.

The model reads Column 0 → Column 1 → Column 2 → ... and predicts the next
column. Each column is one position in the transformer sequence.

## The Strategy: Chord = Column = Position

### Core Idea

Each chord is a **column of the piano roll**. Each column is one position in
the GPT-2 sequence. The model reads columns left to right and predicts the
next column.

```
Song:  START → col_0 → col_1 → col_2 → ... → col_53 → END
                 ↓        ↓        ↓                ↓
Eigen:         (α,β,γ,D) (α,β,γ,D) (α,β,γ,D)    (α,β,γ,D)
```

### What is a Column?

A column is the MIDI data at one time position. It contains:

1. **Pitch activation vector** — length 173 (the active range, step 164-336).
   Each position holds the velocity (0.0 = inactive, normalized vel = active).
   This is the piano roll slice. The topology is physical — adjacent cells
   are adjacent pitches, 53 cells apart = one octave.

2. **Duration** — one scalar, normalized. Shared by all notes in the chord
   (as in the current dataset — all notes in a chord have the same duration).

Total column vector: **174 dimensions** (173 pitch-velocity + 1 duration).

### Input Projection

```
column_vector (174) → Linear(174, n_embd) → chord_embedding (384)
```

One linear layer projects the full column into embedding space. The entire
chord enters the transformer as ONE embedding vector at ONE position.

### Positional Encoding: EigenSpace

No sequential position (wpe). The EigenSpace (α, β, γ, D) computed from the
chord IS the positional encoding:

```
eigenspace (4) → MLP(4 → 64 → 64 → 384) → positional_embedding (384)
```

The transformer input at each position:
```
x = chord_embedding + eigenspace_positional_embedding
```

### Output Head

The output head predicts the next column:

```
transformer_output (384) → Linear(384, 174) → predicted_column (174)
```

- **Pitch positions**: sigmoid → values in [0, 1]. Interpret as velocity.
  During generation, threshold (e.g., top-5 activations = the chord notes).
- **Duration**: sigmoid or softmax over duration bins.

### Loss Function

NOT cross-entropy on token IDs. This is regression/multi-label:

- **Pitch activations**: Binary cross-entropy per position. The target is
  the column vector (0.0 for inactive, velocity/127 for active).
  Alternatively: MSE on the full vector.
- **Duration**: Cross-entropy over discretized duration bins, or MSE on
  normalized duration.

### Sequence Dimensions

| Metric | Old (token-level) | New (chord-level) |
|--------|-------------------|-------------------|
| Positions per song | ~761 | ~54 |
| Embedding dimension | 384 | 384 |
| Input vocabulary | 345 discrete tokens | 174-dim continuous vector |
| Output | softmax over 345 | sigmoid over 174 |
| EigenSpace alignment | smeared across ~12 tokens | 1:1 per position |
| block_size needed | 1024 | 128 (generous) |

### Generation Process

1. Start with START token (a learned embedding, or a zero column).
2. For each step:
   a. The model predicts a 174-dim vector.
   b. Extract pitch activations: take top-k (k=5 typically), or threshold.
   c. Extract duration from the last dimension.
   d. Compute EigenSpace from the predicted chord (the 53-TET steps +
      their frequency ratios → α, β, γ, D).
   e. Use that EigenSpace as the positional encoding for this new position.
   f. Feed back, predict next column.
3. Stop when END is predicted (a special dimension, or when a learned
   "end probability" crosses threshold).

### Key Properties

- **The chord is atomic**: no serialization, no CHORD_START/CHORD_END needed.
- **Pitch topology is preserved**: cell 0 and cell 1 are one 53-TET step
  apart. Cell 0 and cell 53 are an octave. The model can learn spatial
  patterns (voicing shapes) through the linear projection weights.
- **EigenSpace maps 1:1**: one chord = one eigenspace = one position. No
  chicken-and-egg. No smearing.
- **Sequence is short**: 54 chords vs 761 tokens. Attention cost drops by
  ~200x (quadratic in sequence length).
- **No fake tokens**: no P_212, no V_6, no CHORD_START. The data is the data.

### What Changes in the Codebase

| Component | Current | New |
|-----------|---------|-----|
| `05_midi_mpe_tokenization.py` | Produces flat token sequences | Produces column vectors (pitch-velocity + duration) |
| `06_preprocess_dual_channel.py` | Expands eigenspace per token | Eigenspace already per chord — no expansion needed |
| `08_tokenize_for_training.py` | Writes uint16 token IDs + float16 eigen | Writes float32 column vectors + float32 eigen |
| `09_train_gpt2.py` | Token embedding + flat softmax | Linear input projection + sigmoid/regression output |
| `10_generate.py` | Autoregressive token prediction | Autoregressive column prediction + top-k extraction |
| `eigenspace.py` | EigenSpaceEmbedding MLP | Same — unchanged |

### Relationship to MidiTok (MIR Reference)

MidiTok serializes MIDI events into discrete tokens. ANIMA v3 does not
serialize — it projects the piano roll column directly. This is a different
paradigm:

- MidiTok = NLP approach (discrete tokens, vocabulary, cross-entropy)
- ANIMA v3 = Vision approach (continuous vectors, spatial structure, regression)

Both use transformers. MidiTok follows the language model paradigm applied to
music. ANIMA v3 follows the image generation paradigm applied to the piano
roll. The EigenSpace positional encoding has no equivalent in MidiTok.

ANIMA v3 is comparable to MidiTok in the sense that both are autoregressive
transformer models for music. It is an improved version in the sense that:
1. Chord structure is preserved, not serialized.
2. Pitch topology is explicit, not lost in arbitrary token IDs.
3. Harmonic geometry (EigenSpace) informs the positional encoding.
4. The representation is native to 53-TET — not an adapter on top of 12-TET.

### Open Questions

1. **Velocity encoding in the column**: raw MIDI velocity normalized to [0,1]?
   Or binary (on/off) with a separate velocity channel?
2. **Duration encoding**: one shared duration per chord works for this dataset.
   If future data has per-note durations, the column needs expansion.
3. **END detection**: add a 175th dimension as "end probability"? Or a
   separate classification head?
4. **Training loss weighting**: pitch activations are 98% zeros (sparse).
   Need focal loss or class weighting to avoid trivial "predict all zeros."
5. **How many pitch positions**: use the full 319 (step 106-424) or trim to
   the observed 173 (step 164-336)? Trimming saves compute but limits
   transposition range.
