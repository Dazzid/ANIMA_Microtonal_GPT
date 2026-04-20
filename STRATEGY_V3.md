# ANIMA Microtonal GPT — Strategy V3

Single source of truth for the current phase.
Contains: the goal, the data contract, the tokenization spec, the model-input
plan, the log of decisions and past failures, and the road to follow.
This document also serves as the paper outline.

> **Current state (2026-04-17, updated live).** Phase A tasks 1–6 complete.
> Task 7 active: training `modelA_hybrid_v1` with the Large preset
> (L12, H12, E768, ~110 M params). Launched from terminal — user monitors
> progress directly. See §9 for the per-task log and §13.2 stage D for the
> canonical launch command.

---

## 1. Mission

Train a GPT that generates **harmonic chord progressions in 53-TET
microtonality** with learned voicings, conditioned on:

- Transformation **Type** (14 classes, from folder name)
- **Style** (16 canonical buckets)
- **Form** (A/B/C/D, intro, head, coda, segno; with `|:` `:|` repeats)
- **Tonality** (key)
- **Time** (per-chord duration + plain `BAR` marker)

Two models will be compared:

- **Model A — Hybrid Symbolic (L1) + MIDI/MPE (L2)** — this phase
- **Model B — Hybrid Symbolic (L1) + full 53-TET holdrian column (L2)** — next phase

A is trained first. When A converges, B is trained on the same L1 with a
different L2 and the two are compared on identical prompts.

---

## 2. Dataset (built, verified)

Parallel twin-tree, 1:1 mirror:

```
dataset/midi_files/53_tet_mpe/<type_dir>/<stem>.mid
dataset/text_files/53_tet_mpe/<type_dir>/<stem>.txt
```

Verified counts (April 2026 build):

- MIDI files: **672,840**
- Text files: **672,840**
- Type subdirs (14): `type_0_major`, `type_0_minor`, `type_1_minor`,
  `type_1_neutral`, `type_2_minor`, `type_2_subminor`, `type_3_major`,
  `type_3_minor`, `type_4_minor`, `type_4_upmajor`, `type_5_major_v2`,
  `type_5_minor`, `type_6_minor`, `type_6_neutral_n`
- Stem alignment between MIDI and TXT trees confirmed (empty `diff`)

Text sidecar contains everything MIDI cannot express:
`<style>`, style name, `<tonality>`, key, `Form_*`, `|:`, `:|`, `|`, and per
chord `. <duration> <root> <quality> <extensions...> [/ <bass>]`.

Example head of a sidecar:
```
<style> Latin Form_A |: . 4.0 C maj7 | . 4.0 A m7 | ...  :|
```

Builder: `src/generate_53tet_dataset.py`.

---

## 3. Approach: Option 2 — Hybrid L1 + L2

Per chord, emit two aligned blocks in one stream:

```
[ L1 symbolic tokens from .txt ]   [ L2 MIDI/MPE tokens from .mid ]
```

- **L1** = semantic label (what chord in what form in what style).
- **L2** = voicing (which 53-TET pitches, durations, velocities).
- **Alignment**: k-th chord block in the `.txt` ↔ k-th `CHORD_START…CHORD_END`
  block in the `.mid`.
- **Song-level L1 tokens** (type, style, tonality) appear once at the start.
- **Structural L1 tokens** (`|`, `|:`, `:|`, `Form_*`) are emitted at their bar
  boundaries between chord blocks.
- **One vocabulary, one autoregressive loss, one head.** No dual decoder.

### Why this layout

- Voicings are the research target → MIDI/MPE must stay in the stream.
- The previous 12-TET model had no form signal; text sidecars fix that.
- Making every chord self-describing (symbol in L1, realization in L2)
  lets the model learn `symbol → voicing` directly and generate either
  from scratch or conditioned on a partial symbolic prompt.

### Worked example (2 chords)

Raw text sidecar fragment:
```
<style> Jazz <tonality> C_major TYPE_0_major Form_A |: . 4.0 C maj7 | . 4.0 A m7 :|
```

Matching MIDI has two chord blocks. Interleaved stream the model sees:
```
<start>
STYLE_Jazz TONALITY_C_major TYPE_0_major FORM_A |:
. 4.0 C maj7                                         ← L1 block 1
CHORD_START DUR_4.0 P_212 V_3 P_243 V_3 P_265 V_2 P_284 V_3 P_306 V_2 CHORD_END   ← L2 block 1
BAR
. 4.0 A m7                                           ← L1 block 2
CHORD_START DUR_4.0 P_209 V_3 P_240 V_3 P_262 V_2 P_281 V_3 P_303 V_2 CHORD_END   ← L2 block 2
:|
<end>
```

---

## 4. Level 1 vocabulary (symbolic, from `.txt`)

Canonical source files: `src/formats.py` (style / structural normalization)
and `src/convention.py` (53-TET chord naming rules, 104 canonical symbols).

Counts below come from a full scan of all 672,840 sidecars (see
`dataset/l1_alphabet.json` for the raw histograms and companion
`scripts/scan_l1_alphabet.py` logic).

| Group        | Count | Values                                                    | Source |
|--------------|------:|-----------------------------------------------------------|--------|
| Header       | 3     | `<style>`, `<tonality>`, `<type>`                         | literal |
| Type         | 14    | `TYPE_0_major` … `TYPE_6_neutral_n`                       | folder name |
| Style (raw → canonical) | 131 raw → 16 canonical | `Jazz Blues Folk Bossa Reggae Samba Funk Pop Son Rock Soul Balad RnB Gospel Afoxé "Even 8ths"` | `formats.correctStyleTokensInMeta` |
| Tonality     | ~24   | `C_major`, `A_minor`, …                                   | filename |
| Form         | 9     | `FORM_INTRO A B C D VERSE HEAD CODA SEGNO`                | sidecar |
| Structural   | 5     | `.`, `|`, `|:`, `:|`, `/`                                 | sidecar |
| Duration (L1)| 10    | same grid as L2 `DUR_*` (float literal beats)             | sidecar |
| Root         | **71**| 53-TET root names incl. microtonal prefixes (see below)   | sidecar |
| Quality      | **193** + 1 implicit | 53-TET chord qualities (see below); empty string `''` renamed to `maj_implicit` in vocab | sidecar |
| Extensions   | **16**| joined phrases: `'add 9'`, `'add b9'`, `'alter b5'`, …    | sidecar |
| Slash bases  | 64    | subset of Root set                                        | sidecar |

### 4.1 Root alphabet (71)

Bare 12-TET letters, `^` / `^^` (up / double-up), `v` / `vv` (down / double-down),
plus rare `##` / `bb`. Exhaustive list:

```
A, A##, Abb, B, B#, B##, Bb, Bbb, C, C#, C##, Cb, Cbb,
D, D#, D##, Dbb, E, E#, E##, Ebb, F, F#, F##, Fb, Fbb,
G, G#, G##, Gbb,
^A, ^B, ^Bb, ^C, ^C#, ^D, ^E, ^F, ^F#, ^G, ^G#,
^^A, ^^B, ^^Bb, ^^C, ^^C#, ^^D, ^^E, ^^Eb, ^^F, ^^F#, ^^G,
vA, vB, vBb, vC, vC#, vD, vD#, vE, vF, vF#, vG, vG#,
vvA, vvB, vvC#, vvD#, vvE, vvF#, vvG#
```

### 4.2 Quality alphabet (193)

Union of `convention.py` (104 canonical triad+7 combinations) and legacy
12-TET labels surviving in the sidecars (`maj7`, `m7`, `dom7`, `ø7`, `o7`,
`dim7`, `aug`, `sus4`, `sus7`, `power`, `o`, `m`, `vM`, `Nm`, …) plus
rare fallback labels `(step34)`, `(step36)`, `(step37)`, `(step38)` and
`N.C.` (no-chord).

Full list dumped to `dataset/l1_alphabet.json` → `qualities`. The empty
string `''` (6.47M occurrences = implicit major triad) is stored in the
vocabulary as the explicit token `maj_implicit`.

### 4.3 Extension alphabet (16, joined phrases)

```
add 2, add 7, add 9, add 11, add 13, add b6, add b9, add b13,
add #7, add #9, add #11, alter b5, alter #5, alter #9, alter #11,
sus7
```

Each is **one** token with the space preserved.

### Rules

- Style is chosen by the substring rules in `formats.correctStyleTokensInMeta`. **Diversity preserved (16 buckets, not collapsed to 4).**
- Root in L1 is a 53-TET **name string** (letter + optional `^`/`v` prefix). L2 still carries absolute pitch as `P_<step>`.
- `BAR` is a plain structural marker. **No numbering.**
- Quality `'N.C.'` and `(stepNN)` are kept as-is (low frequency). Flagged for Phase B review.

---

## 5. Level 2 vocabulary (MIDI/MPE, from `.mid`)

Extracted from `src/tokenizer.py` (current state, **vocab_size = 2930**):

| Group       | Count | Values                                                        |
|-------------|------:|---------------------------------------------------------------|
| Special     | 4     | `<pad> <start> <end> <sep>`                                   |
| Structural  | 4     | `CHORD_START`, `CHORD_END`, `BAR`, `REST`                     |
| Duration    | 10    | `DUR_` ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0} |
| Pitch       | 319   | `P_106` … `P_424` (absolute 53-TET step; octave 2 … octave 8) |
| Velocity    | 8     | `V_1` … `V_8`                                                 |

Per-chord block:
```
CHORD_START  DUR_<d>  P_<step> V_<v>  P_<step> V_<v>  …  CHORD_END
```
Max 8 notes per chord (`MAX_CHORD_NOTES = 8`).

Known stray — **removed 2026-04-17**: `ROOT_<pc>` tokens (53 of them) have
been deleted from vocab and encoder. Decoder already ignored them. Vocab
dropped 2662 → 2609.

Implementation: `MPETokenizer.encode_chords` / `chords_to_midi` in
`src/tokenizer.py`.

---

## 6. EigenSpace — positional embedding, not a token

Each chord has a 4-vector `(α, β, γ, D)` from `src/eigenspace.py`
(`D` = dissonance; previously called `δ` and mis-labeled "Plomp–Levelt").
It is **not tokenized**.

Pipeline:
1. Compute per-chord `(α, β, γ, D)` from the **L1 symbolic root** (not
   `min(pitches)` — slash chords must honour their named bass). Saved as
   `<stem>.eigen.npy`, shape `(N_chords, 4)` float32.
2. Expand to per-token via the spans returned by
   `l1.build_merged_tokens_with_spans` under **Visibility Rule B** (locked):
   - Header tokens (`<start>`, `STYLE_*`, `TONALITY_*`, `TYPE_*`) → span `-1`
     → default `[1.0, 1.0, 2.0, 0.0]` = `HEADER_EIGEN`.
   - L1 chord `k` and L2 chord `k` share span `k` → `eigen_chord[k]`.
   - Structural markers (`|`, `|:`, `:|`, `FORM_*`) between chord `k-1` and
     chord `k` carry span `k-1`: **the previous chord's eigenspace persists
     across the gap; no future leakage.**
   - `<end>` inherits the last chord; padding uses `HEADER_EIGEN`.
3. Project `(4 → n_embd)` via `EigenSpacePositionalEncoding` (small MLP)
   and **add** to the positional embedding before the transformer.

---

## 7. Decision log (locked)

Do not revisit without cause.

- **Option 2 is Phase A.** Option 1 (symbolic-only) is skipped. Option 3
  (holdrian column) is Phase B.
- **16 style buckets**, sourced from `formats.correctStyleTokensInMeta`.
  Never collapsed.
- **BAR is plain**, no numbering. Structure comes from `Form_*` + `|` +
  `|:` `:|` + EigenSpace continuity.
- **Root in L1 is a letter.** 53-TET realization is L2 only.
- **EigenSpace is positional**, not lexical. One autoregressive head.
- **`src/formats.py` is canonical** for style buckets, note names, chord
  qualities, and structural markers.
- **Previous 12-TET paper is a reference, not a spec.** Concepts transfer;
  vocabularies do not.
- **Method B vocabulary = flat compound `H_<step>_<vel>`** (2026-04-20,
  publication-deadline decision). True-column / multi-hot chord emission
  deferred to post-deadline. Rationale in §B.1.

---

## 8. Failure log (what went wrong, what we learned)

Record failures here so future sessions do not repeat them.

| # | Failure | Root cause | Fix / lesson |
|---|---------|------------|--------------|
| F1 | Previous 12-TET model produced harmony but ignored form | No form information in training input | Text sidecar now emits `Form_*`, `|:`, `:|` inline |
| F2 | Agent proposed collapsing styles to 4 buckets | Optimized for vocab size without checking project conventions | `formats.py` is canonical; 16 buckets stay |
| F3 | `BAR_1 … BAR_64` tokens added | Assumed absolute bar indices help form | Musicians think in sections, not bar numbers; BAR is now plain. Absolute indices fight generalization across repeated sections |
| F4 | `ROOT_<pc_0..52>` invented from MIDI bass | Mixed L1 semantics into L2 | Root is a letter in L1 only; remove `ROOT_*` from vocab |
| F5 | Agent treated the published 12-TET paper as the current spec | Skipped alignment step | Align on intent before code. Previous paper = reference, not spec |
| F6 | Earlier drafts wrote strategy as opinionated prose | Wrong register | Strategy is instructional: state decisions, not opinions |
| F7 | Initial L1 vocab assumed `formats.getNotes()` (20) / `getNatures()` (18) as the alphabet | Those tables describe **12-TET input** to the transformer, not the **53-TET sidecar output** | Scan the actual sidecars. Truth is 71 roots, 193 qualities, 16 ext phrases — pulled from `dataset/l1_alphabet.json` |
| F8 | `pack_data_v3.py` crashed at the very end of a ~7 min run, leaving stale `meta.json` | Final step cast the 21.6 GB fp16 `train_eigen` to fp32 (~43 GB) just to compute per-dim mean/std — OOM | Stream stats in 8192-seq chunks (two-pass mean/variance in float64). Added `--regen-meta-only` so we don't have to re-pack when only meta is missing. Always free big arrays before downstream reductions |

---

## 9. Open tasks (the road)

### Phase A — Hybrid Symbolic + MIDI/MPE (current)

1. **Tokenizer cleanup** ✅ 2026-04-17
   - `ROOT_<pc>` tokens removed; vocab 2662 → 2609
   - Encoder no longer emits ROOT; decoder skips legacy ROOT tokens
2. **L1 parser** ✅ 2026-04-17
   - `parse_text_sidecar(path) -> L1Parsed` in `src/l1.py`
   - Header tokens once, then per-chord `ChordEvent` / `StructuralEvent` stream
   - Uses alphabet frozen in `dataset/l1_alphabet.json`; empty quality → `maj_implicit`
   - DUR quantization in place; style coverage 99.65 %
3. **L1+L2 interleaver** ✅ 2026-04-17
   - `merge_levels(parsed, l2_tokens)` + `build_merged_tokens(midi, txt, tok)` in `src/l1.py`
   - Aligns by chord index; emits `<start> …header… [L1_k L2_k | structurals]* <end>`
   - Asserts equal chord counts; raises on mismatch
   - Validated on 300/300 paired songs
4. **EigenSpace sidecar builder** ✅ 2026-04-17
   - `src/eigen_sidecar.py` + parallel CLI `src/build_eigen_sidecars.py`
   - Per chord computes `(α, β, γ, D)` from L1 symbolic root (root-invariant;
     slash chords use named root, not `min(pitches)`)
   - `classify_intervals` rewritten to return a 3-tuple (α, β, γ); dissonance D
     comes from `DissonanceMap` lookup (renamed from the Plomp–Levelt misnomer)
   - Full sweep: **672,518 / 672,840 = 99.95 %** `.eigen.npy` files written
     in 7.3 min @ 1 540 files/s
   - The 322 failures cluster in 23 base songs × 14 key transpositions — all
     L1/L2 chord-count mismatches from an upstream XML generator bug.
     **Decision: skip these songs; no further investigation for Phase A.**
   - Total: 84 969 402 chord rows on disk
5. **Model update** ✅ 2026-04-17
   - `EigenSpacePositionalEncoding` MLP in `src/model.py`
   - 4-D chord vector projected and summed into the positional embedding
   - `vocab_size = 2930`, `block_size = 4096`, `n_eigen = 4` (α, β, γ, D)
   - Forward + loss smoke-tested on packed tensors
6. **Preprocess + pack v3** ✅ 2026-04-17
   - Old two-stage pipeline (MIDI → per-song JSON → .bin) retired
   - New single-stage packer: `src/pack_data_v3.py` takes paired
     `(.mid, .txt, .eigen.npy)` directly to `train_tokens.bin` / `train_eigen.bin`
   - `merge_levels_with_spans()` / `build_merged_tokens_with_spans()` in
     `src/l1.py` returns `(tokens, chord_spans)` under **Visibility Rule B**
   - Outputs `uint16` tokens + `float16` eigen at `dataset/tokenized/`,
     plus `meta.json` (per-dim eigenspace stats over train split) and `vocab.json`
   - **Full dataset packed** from 674 366 paired songs:
     - train: 660 870 seqs × 4097 tokens = 2.71 B train tokens
     - val:    13 496 seqs × 4097 tokens = 55.3 M val tokens
     - vocab_size = **2930**, block_size = **4096**
     - on-disk: 5.4 GB train_tokens, 21.6 GB train_eigen, 110 MB val_tokens, 442 MB val_eigen
   - **Eigenspace z-score stats (train split)** — use these at model input:
     - α: mean 1.0946, std 0.1184
     - β: mean 1.1995, std 0.2430
     - γ: mean 1.8771, std 0.1774
     - D: mean 5.6368, std 6.9195   ← much larger scale than α/β/γ; **must** z-score
   - **Fix (2026-04-17)**: original packer OOM-crashed at the final
     `train_eigen.astype(float32)` step (~43 GB alloc). Replaced with
     `_eigen_stats_streaming()` (8192-seq chunks, two-pass mean/variance
     in float64) and a `--regen-meta-only` flag for rebuilding `meta.json`
     from existing bins without re-packing. See F8 in §8.
7. **Train Model A** ✅ 2026-04-20
   - Run name: `modelA_hybrid_v1`; checkpoints at `checkpoints/modelA_hybrid_v1/`
   - **Preset used: Large (GPT-2-small class, ~90.5 M params)**
     - `n_layer=12`, `n_head=12`, `n_embd=768`, `block_size=4096`,
       `n_eigen=4`, `eigen_hidden=128`, `use_sequential_pos=True`
     - batch 8 × grad-accum 8 = effective batch 64; 524 288 tokens/iter
   - **Completed at iter 49 500 / 50 000**, `best.pt` with
     `best_val_loss = 0.2600`. Checkpoint loads and generates coherent
     L1+L2 streams end-to-end; see `src/10_generate_midi_v2_fresh.ipynb`
     (V3-aware generator, validated 2026-04-20: 208 tok/s, 76 chords /
     1024 tokens, 20 unique eigen vectors at `CHORD_START`, MIDI export OK).
   - Canonical launch command lives in §13.2 stage D.
8. **Evaluate Model A** (§10) — pending. Metrics: form adherence,
   symbol↔voicing consistency, style/type conditioning response, novelty
   vs memorization.

### Phase B — Symbolic 53-TET GPT + deterministic MIDI translator

**Goal.** Compare Model A (hybrid L1 + MIDI/MPE L2) against a second model
that generates a **pure symbolic 53-TET stream** and converts to MIDI only
*after* generation. This isolates the benefit of learning voicings in a
microtonality-native vocabulary instead of as MIDI pitch events.

**Isolation rule.** Phase B introduces **new modules with the `_b` suffix**.
It must not modify, overwrite or reuse in-place any Model A production file
listed in §11. Model A must stay reproducible from its existing modules
alone.

#### B.0 Isolation contract (hard rules, binding)

The primary risk in Phase B is accidentally breaking Model A. The
following rules are non-negotiable; any deviation requires an explicit
decision entry in §7.

**Files that are FROZEN for Phase B — never edited, never renamed:**

- **Code (`src/`):** `tokenizer.py`, `pack_data_v3.py`, `pack_data.py`,
  `generate.py`, `train.py`, `trainer.py`, `model.py`, `preprocess.py`,
  `preprocess_runner.py`, `configurator.py`, `l1.py`, `eigen_sidecar.py`,
  `build_eigen_sidecars.py`, `eigenspace.py`, `formats.py`, `voicing.py`,
  `chord_mapping.py`, `convention.py`, `transposition.py`,
  `xmlTranslator.py`, `generate_53tet_dataset.py`, `midi_viz.py`,
  `play_mpe.py`, `utils.py`, `mingpt_utils.py`,
  `10_generate_midi_v2_fresh.ipynb`.
- **Artifacts:** `dataset/tokenized/` (all `*.bin` + `meta.json` +
  `vocab.json`), `dataset/l1_alphabet.json`, `dataset/text_files/`,
  `dataset/midi_files/`, every `.eigen.npy` sidecar,
  `checkpoints/modelA_hybrid_v1/`, `checkpoints/best.pt`,
  `checkpoints/final.pt`, `checkpoints/latest.pt`.

**Rules:**

1. **New code only in new files** with the `_b` suffix (or new
   directories). No edits to any file listed above.
2. **No in-place edits to shared modules.** If Phase B needs a behaviour
   change in `eigenspace.py`, `chord_mapping.py`, `convention.py`,
   `formats.py`, `l1.py`, etc., we **copy** the needed function into a
   Phase-B module and modify the copy — the original stays untouched.
   Read-only `import …` of existing modules is fine.
3. **Separate output trees.** Phase B writes only to
   `dataset/tokenized_b/` and `checkpoints/modelB_column_v1/`. Never to
   `dataset/tokenized/` or to existing checkpoint dirs.
4. **Reuse `.eigen.npy` sidecars read-only.** Phase B consumes them,
   never rewrites them.
5. **Reuse the paired twin-tree (`.mid` + `.txt`) read-only.** No
   re-generation of the dataset in Phase B.
6. **No monkey-patching of Method A classes.** `tokenizer_b.py` is a
   *new* class, not a subclass that overrides methods on `MPETokenizer`.
7. **Git discipline.** Every Phase-B commit touches only `*_b.*` files,
   new notebooks starting with `12_`, `dataset/tokenized_b/`,
   `checkpoints/modelB_column_v1/`, or this strategy document. Anything
   outside that set → stop and ask.
8. **Regression gate before any Phase-B merge.** Re-run
   `src/10_generate_midi_v2_fresh.ipynb` end-to-end and confirm Model A
   behaviour is unchanged: `vocab_size = 2930`,
   `best_val_loss = 0.2600`, successful MIDI export, same
   order-of-magnitude generation speed. If any Method A number drifts →
   the Phase-B change is reverted.
9. **Append-only strategy doc.** Phase B edits only add to §9 Phase B
   and (when needed) a new §5-bis "Level 2 — 53-TET column" section.
   §4, §5, §6, §7, §8 (Method A spec + decision log + failure log) are
   append-only.

#### B.1 Representation — "53-TET column" (LOCKED 2026-04-20)

**Decision (publication deadline).** Option (i) — **flat holdrian stream**
with a compound pitch+velocity token. Architecture, block size, chord-block
protocol, and total vocabulary count stay **identical to Method A** so the
A↔B comparison isolates exactly one variable: the *interpretation* of the
pitch token (MIDI-derived `PV_` vs pure holdrian `H_`). True-column /
multi-hot chord emission (option ii) is deferred to a post-deadline phase.

**Per-chord block (same shape as Method A):**

```
CHORD_START  DUR_<d>  H_<step>_<vel>  H_<step>_<vel>  …  CHORD_END
```

- `H_<step>_<vel>` — compound **holdrian comma + velocity bin**.
  - `step` ∈ `[PITCH_OFFSET_B .. MAX_53TET_STEP_B] = [106 .. 424]` (319
    values), **absolute 53-EDO step index**, octave-agnostic in naming.
  - `vel` ∈ `[1..8]` (8 bins), same grid as Method A `V_<v>`.
  - Total H_ tokens: `319 × 8 = 2552`.
- `H_` is **not** a MIDI derivative. The paper claim: the vocabulary has
  **zero MIDI semantics**; generation emits a pure 53-EDO symbolic stream;
  MIDI only appears at inference time via the deterministic translator
  `src/holdrian_to_midi.py` (task B-4).
- Notes inside a chord are ordered **low → high by `step`**, same as A.
- `MAX_CHORD_NOTES = 8` unchanged. `DUR_*` grid unchanged.
- L1 block (`. DUR R_ Q_ X_ /`), `CHORD_START`, `CHORD_END`, `BAR`,
  `REST`, `TYPE_*`, `STYLE_*`, `TONALITY_*`, `FORM_*`, specials: all
  **reused unchanged** from Method A via `l1.load_alphabet()`.
- EigenSpace `(α, β, γ, D)` reused **unchanged** — same MLP, same
  Visibility Rule B, same `.eigen.npy` sidecars. `eigenspace.py` already
  operates on 53-EDO step integers; no change needed.

**Final vocabulary count (Method B):**

| Group       | Count | Notes                                                      |
|-------------|------:|------------------------------------------------------------|
| Special     | 4     | `<pad> <start> <end> <sep>`                                |
| Structural  | 4     | `CHORD_START CHORD_END BAR REST`                           |
| Duration    | 10    | `DUR_*` (same grid as A)                                   |
| Type        | 14    | `TYPE_*` (reused)                                          |
| Style       | 16    | `STYLE_*` (reused)                                         |
| Form        | 9     | `FORM_*` (reused)                                          |
| Holdrian+Vel| 2552  | `H_<step>_<vel>` — **replaces** A's `PV_<step>_<vel>`      |
| L1 (via `l1.load_alphabet`) | ~321 | R_, Q_, X_, TONALITY_, L1 structural (`.`, `|`, `|:`, `:|`, `/`) |
| **Total**   | **~2930** | **Matches Method A so `model.py` loads with no change.** |

The exact total is whatever `tokenizer_b.TokenizerB._build_vocab` emits and
is written to `dataset/tokenized_b/vocab.json` at pack time. The target is
to land at the same `vocab_size = 2930` as Method A; any drift is recorded
here and does not break the isolation contract (B.0) since Method A's own
`vocab.json` stays untouched.

**Why this shape (and not a 3-way `H_<step>_<oct>_<vel>` or a multi-hot
column):**

1. Same seq-length budget as A → identical `block_size = 4096` →
   apples-to-apples val-loss comparison.
2. Same AR softmax head → reuse `model.py` verbatim (isolation contract
   rule 1, 2, 6).
3. `step` is **already** absolute across octaves (range 106..424 spans
   octaves 2..8). Octave information is implicit in the integer.
   Splitting it out adds tokens without adding expressiveness.
4. Multi-hot / true-column (option ii) requires a new loss head and a new
   EigenSpace expansion, violating the deadline constraint and diluting
   the research claim. Scheduled for post-deadline follow-up.

#### B.2 Post-hoc MIDI translator

A deterministic function `holdrian_to_mpe_midi(tokens, path)` takes the
generated stream and writes MPE MIDI using the same RPN pitch-bend setup
as `tokenizer.chords_to_midi`. No learning at this step — only
`step,oct → (nearest MIDI note, bend)` using the existing 53-EDO tables
in `chord_mapping.py` / `convention.py`.

#### B.3 New files (no collision with Model A)

| New file                         | Role                                                         |
|----------------------------------|--------------------------------------------------------------|
| `src/tokenizer_b.py`             | 53-TET symbolic tokenizer (H_ tokens, L1 reused)             |
| `src/pack_data_b.py`             | Packs `(.mid, .txt, .eigen.npy)` → `dataset/tokenized_b/*`   |
| `src/holdrian_to_midi.py`        | Deterministic 53-TET-column → MPE MIDI translator            |
| `src/generate_b.py`              | Inference entry for Model B (mirrors `generate.py`)          |
| `src/train_b.py`                 | Training entry for Model B (mirrors `train.py`)              |
| `src/12_generate_modelB.ipynb`   | Generation notebook for Model B, validated end-to-end        |
| `dataset/tokenized_b/`           | Packed bins + `meta.json` + `vocab.json` for Model B         |
| `checkpoints/modelB_column_v1/`  | Checkpoints for Model B                                      |

Model A modules (`tokenizer.py`, `pack_data_v3.py`, `generate.py`,
`train.py`, `model.py`) are **imported as-is where possible** (e.g.
`model.py` already parameterises `vocab_size`; no fork needed) and
**never edited** during Phase B.

Shared, read-only dependencies: `formats.py`, `l1.py`, `eigenspace.py`,
`eigen_sidecar.py`, `chord_mapping.py`, `convention.py`, `voicing.py`,
`transposition.py`, `generate_53tet_dataset.py`.

#### B.4 Task list

1. **B-1 Freeze the 53-TET column vocabulary.** ✅ 2026-04-20 — locked
   to flat compound `H_<step>_<vel>` (see §B.1). Implement
   `src/tokenizer_b.py` with encode/decode + round-trip test on 100 songs.
2. **B-2 Port the packer.** Write `pack_data_b.py` (copy of
   `pack_data_v3.py` re-targeted to `tokenizer_b`). Produce
   `dataset/tokenized_b/{train,val}_{tokens,eigen}.bin`, `meta.json`,
   `vocab.json`. EigenSpace reuses the existing `.eigen.npy` sidecars
   unchanged.
3. **B-3 Train `modelB_column_v1`.** Same preset as Model A
   (L12 H12 E768, block_size 4096, effective batch 64, 50 k iters) for a
   clean A↔B comparison.
4. **B-4 Translator.** Implement `holdrian_to_midi.py` and verify
   round-trip on a 100-song sample of the training data: pack → decode →
   translate → re-parse the resulting MIDI → pitch-class histogram
   matches original to within rounding.
5. **B-5 Generation notebook.** `src/12_generate_modelB.ipynb`, same
   conditioning UX as notebook 10 but using `generate_b.py` + the
   translator. Must produce audible MPE MIDI.
6. **B-6 Evaluate Model B** against the metrics in §10, on **the same
   prompts** used for Model A.
7. **B-7 A vs B comparison.** Same prompts, same seeds where possible.
   Report: val-loss, form adherence, symbol↔voicing consistency,
   microtonal pitch-class entropy per `TYPE_*`, novelty vs training.

### Phase C — Cleanup

Delete or archive the obsolete files listed in §11 once A and B are
reproducible from the production modules alone.

---

## 10. Evaluation criteria

Model A success is judged on:

1. **Form adherence.** Given a prompt with `FORM_A |: …` the generation
   must emit a matching `:|` and a plausible `FORM_B` continuation.
   Metric: fraction of generations with balanced `|:`/`:|` and a detected
   section change.
2. **Symbol↔voicing consistency.** Per chord block, the L2 pitches should
   spell the L1 chord symbol within a 53-TET tolerance.
   Metric: fraction of chords whose L2 pitch set maps (via
   `voicing.py` / `chord_mapping.py`) back to the emitted L1 symbol.
3. **Style conditioning.** Fix type, vary style; generate N progressions.
   Style label must be recognizable in the L1 token statistics and in
   listening tests.
4. **Type conditioning.** Fix style, vary type; L2 pitch histograms must
   shift between types.
5. **Novelty vs memorization.** n-gram overlap with training set on L1
   and on L2 below a threshold (TBD).

---

## 11. Main files in `src/`

### Production — keep, extend
| File | Role |
|------|------|
| `tokenizer.py`              | L1+L2 tokenizer, `MPETokenizer`, dataset, round-trip |
| `formats.py`                | Canonical style/note/quality/structural definitions |
| `xmlTranslator.py`          | iRealXML → chord + form sequence |
| `voicing.py`                | Chord symbol → voicing realization |
| `eigenspace.py`             | Per-chord `(α, β, γ, D)` computation |
| `chord_mapping.py`          | 53-TET note names, chord interval maps |
| `convention.py`             | 53-TET chord naming convention |
| `transposition.py`          | Key/root transposition helpers |
| `generate_53tet_dataset.py` | Builder of the parallel twin-tree dataset |
| `preprocess.py`             | Training-data preparation |
| `preprocess_runner.py`      | Parallel driver for preprocessing |
| `pack_data.py`              | Pack preprocessed JSON → memory-mapped bins |
| `model.py`                  | GPT-2 model (will gain EigenSpace PE) |
| `trainer.py`                | Generic training loop |
| `train.py`                  | Training entry point |
| `configurator.py`           | Config override mechanism |
| `generate.py`               | Inference / sampling entry |
| `midi_viz.py`               | MIDI visualization (Plotly) |
| `play_mpe.py`               | Playback of MPE MIDI |
| `utils.py`, `mingpt_utils.py` | Shared helpers |

### Obsolete — flag, clean in Phase C
| File | Notes |
|------|-------|
| `build_parallel_dataset.py`    | Superseded by `generate_53tet_dataset.py` |
| `04_runner.py`                 | Old training runner |
| `debug_run.py`                 | Ad-hoc debug script |
| `test_metadata_export.py`      | One-off metadata dump |
| `test_something_standalone.py` | Scratch |
| `to_compare.py`                | Scratch |
| `utils_lenghts.py`             | Dead constants |

### Drifted dev notebooks — read-only, clean in Phase C
`01_musicXML_parser.ipynb`, `02_EigenSpace_mapping.ipynb`,
`03_map_MIDI_to_EigenSpace.ipynb`, `03_map_MIDI_to_FTT.ipynb`,
`04_53TET_conversion.ipynb`, `04_test_mpe.ipynb`,
`04.5_data_augmentation_53edo.ipynb`, `05_generation.ipynb`,
`05_mpe_tokenizer.ipynb`, `05_play_mpe53.ipynb`,
`05.5_tokenization_quality_check.ipynb`, `06_plots_and_figures.ipynb`,
`10_generate_midi_v2_fresh.ipynb`, `11_dataset_qc.ipynb`,
`MIDI_test.ipynb`, `run.ipynb`, `testing_something.ipynb`

---

## 12. Agent workflow (process rules)

Earlier sessions drifted. These rules prevent repeat drift.

1. **Align on intent before touching code.** If the spec is ambiguous, ask.
2. **`src/formats.py` is canonical.** Don't reinvent tables elsewhere.
3. **Preserve diversity.** Don't collapse vocabulary categories without
   explicit approval.
4. **Don't invent tokens** that are not backed by sidecar data.
5. **Previous 12-TET paper = reference, not spec.** Concepts transfer;
   vocabularies do not.
6. **One change at a time, verified.** After each tokenizer edit,
   round-trip a sample and confirm vocab size.
7. **Redundant files are flagged, not deleted.** Cleanup is Phase C.
8. **Update this document when a decision changes.** Add to §7 (locked)
   or §8 (failure log) as appropriate.
# ANIMA Microtonal GPT — Strategy V3

This document is the single source of truth for the current phase of the project.
It defines the goal, the approach, the data contract, the tokenization, the
model input, the agent workflow, and the role of each file in `src/`.
It is also the outline of the paper.

---

## 1. Mission

Train a GPT that generates **harmonic chord progressions in 53-TET
microtonality** with learned voicings, conditioned on:

- **Type** of microtonal transformation (from folder name, 14 classes)
- **Style** (jazz, bossa, rock, soul, …)
- **Form** (A / B / C / D, intro, head, coda, repeats)
- **Tonality** (key)
- **Time structure** (bars, chord durations)

Two concrete models will be compared:

- **Model A — Hybrid Symbolic + MIDI/MPE** (this phase)
- **Model B — Hybrid Symbolic + full 53-TET holdrian column** (next phase)

We start with A. When A works, we train B and compare.

---

## 2. Dataset (done)

Parallel twin-tree 53-TET dataset, one-to-one mirror:

```
dataset/midi_files/53_tet_mpe/type_<k>_<label>/<stem>_<type>.mid
dataset/text_files/53_tet_mpe/type_<k>_<label>/<stem>_<type>.txt
```

- 14 type subdirs (microtonal transformations, including `type_0_major` = identity)
- 672,840 MIDI files + 672,840 matching text files
- Stems are aligned 1:1; the `type_<k>_<label>` folder carries the transformation label

Text sidecars contain the symbolic content previously missing:
`<style>`, style name, `<tonality>`, key, `Form_A/B/C/D`, `|:`, `:|`, `|`,
and per-chord `. <duration> <root> <quality> <extensions...> [/ <bass>]`.

Example (truncated):
```
<style> Latin Form_A |: . 4.0 C maj7 | . 4.0 A m7 | ...  :|
```

---

## 3. Approach: Option 2 — Hybrid L1 + L2

Per chord, we emit two aligned blocks in a single stream:

```
[ L1 symbolic tokens from .txt ]  [ L2 MIDI/MPE tokens from .mid ]
```

- **L1** gives the model the semantic label (what chord, in what form, in what style).
- **L2** gives the model the voicing (which 53-TET pitches, with which durations and velocities).
- Alignment is **by chord index**: the k-th chord block in the `.txt` matches
  the k-th CHORD_START…CHORD_END block in the `.mid`.
- Song-level L1 tokens (style, tonality) appear once at the start.
- Structural L1 tokens (`|`, `|:`, `:|`, `Form_*`) are emitted at their bar
  boundary between chord blocks.

This keeps every chord self-descriptive and lets the model learn the mapping
`symbol → voicing` directly.

### Why this layout

- The previous 12-TET model learned progressions fine but had no form.
  The text sidecar now carries form. This is the fix.
- Voicings are the research target; we must keep MIDI/MPE in the stream.
- L1 and L2 share one vocabulary and one autoregressive loss. No dual head.

---

## 4. Level 1 — Symbolic tokens (from `.txt`)

Canonical source: `src/formats.py`. Do not invent new categories; reuse
`correctStyleTokensInMeta`, `getNotes`, `getNatures`, `listToIgnore`,
`splitChordTokens`, `splitSlashChords`.

Token groups:

| Group         | Examples                                               | Source |
|---------------|--------------------------------------------------------|--------|
| Header        | `<style>`, `<tonality>`, `<type>`                      | literal |
| Style (16)    | `Jazz Blues Folk Bossa Reggae Samba Funk Pop Son Rock Soul Balad RnB Gospel Afoxé "Even 8ths"` | `formats.correctStyleTokensInMeta` |
| Tonality      | `C_major`, `A_minor`, …                                | sidecar |
| Type (14)     | `TYPE_0_major`, `TYPE_1_…`, …                          | folder name |
| Form          | `Form_A Form_B Form_C Form_D INTRO HEAD VERSE CODA SEGNO` | sidecar |
| Structural    | `|`  `|:`  `:|`                                        | sidecar |
| Chord start   | `.`                                                    | sidecar |
| Duration      | `4.0`, `2.0`, `1.0`, `0.5`, …                          | sidecar |
| Root (letter) | `C D E F G A B` + accidentals (20 names from `getNotes()`) | sidecar |
| Quality       | 18 qualities from `getNatures()`                       | sidecar |
| Extensions    | `b9`, `#11`, `add6`, …                                 | sidecar |
| Slash bass    | `/` followed by root letter                            | sidecar |

Rules:
- Style bucket is chosen by the substring rules in `formats.correctStyleTokensInMeta`. Diversity is preserved (16 buckets, not 4).
- Root is a **letter name**, not a pitch class. The 53-TET realization lives in L2.
- `BAR` is a plain structural marker (no numbering). The model learns structure from `Form_*` + `|` + `|:` `:|`.

---

## 5. Level 2 — MIDI/MPE tokens (from `.mid`)

Per chord block:

```
CHORD_START DUR_<d> P_<step0> V_<v0> P_<step1> V_<v1> … CHORD_END
```

- `P_<step>` : 53-TET step index (0..52 within octave, with octave inferred from register band)
- `V_<v>`    : quantized velocity bin
- `DUR_<d>`  : chord duration bin
- `CHORD_START` / `CHORD_END` : block delimiters shared with L1

Implementation lives in `src/tokenizer.py` (`MPETokenizer.encode_chords` /
`chords_to_midi`). No `ROOT_<pc>` tokens — the root lives in L1 as a letter.

---

## 6. EigenSpace — positional embedding, not a token

Each chord has a 4-vector `(α, β, γ, D)` computed by `src/eigenspace.py`.
We do **not** tokenize it.

Pipeline:
1. Compute per-chord `(α, β, γ, D)` → save as `<stem>.eigen.npy` sidecar.
2. At model input, repeat the chord's vector across every token inside that
   chord's L1+L2 block.
3. Project through a small MLP `(4 → n_embd)` and **add** to the positional
   embedding before the transformer blocks.

This injects harmonic geometry as a continuous bias, leaving the discrete
vocabulary untouched.

---

## 7. Model A vs Model B

| Aspect        | Model A (now)                     | Model B (next)                         |
|---------------|-----------------------------------|----------------------------------------|
| L1            | Symbolic text sidecar             | Symbolic text sidecar (same)           |
| L2            | MIDI + MPE                        | 53-TET holdrian column, no MIDI        |
| Output        | MIDI directly                     | Column → MIDI via post-processor       |
| Purpose       | Learn voicings from real MIDI     | Learn pure microtonal spelling         |
| Training      | First                             | After A converges                      |

Checkpoints and run names must include the model letter (`modelA_*`, `modelB_*`)
so comparisons stay unambiguous.

---

## 8. Agent Workflow (process rules)

These rules exist because earlier sessions drifted. Follow them.

1. **Align on intent before touching code.** If the spec is ambiguous, ask. Do not guess.
2. **`src/formats.py` is canonical** for style buckets, note names, chord qualities, and structural markers. Do not reinvent them in other files.
3. **Preserve diversity.** Do not collapse vocabulary categories (styles, qualities, forms) without explicit approval.
4. **Do not invent tokens** that are not backed by data in the sidecars (e.g. no `ROOT_<pc>` from MIDI bass).
5. **The previous 12-TET paper is a reference, not a spec.** Concepts carry over; vocabularies do not.
6. **One change at a time, verified.** After each tokenizer edit, round-trip a sample and confirm vocab size.
7. **Redundant files are flagged, not deleted.** Cleanup happens in its own phase.

---

## 9. Main files in `src/`

### Production — keep, extend
| File | Role |
|------|------|
| `tokenizer.py`              | L1+L2 tokenizer, `MPETokenizer`, dataset class, round-trip |
| `formats.py`                | Canonical style/note/quality/structural definitions |
| `xmlTranslator.py`          | Parse iRealXML → chord + form sequence |
| `voicing.py`                | Chord symbol → voicing realization |
| `eigenspace.py`             | Per-chord `(α, β, γ, D)` computation |
| `chord_mapping.py`          | 53-TET note names, chord interval maps |
| `convention.py`             | 53-TET chord naming convention |
| `transposition.py`          | Key/root transposition helpers |
| `generate_53tet_dataset.py` | Builder of the parallel twin-tree dataset (done) |
| `preprocess.py`             | Dual-channel training-data preparation |
| `preprocess_runner.py`      | Parallel driver for preprocessing |
| `pack_data.py`              | Packs preprocessed JSON into memory-mapped bins |
| `model.py`                  | GPT-2 dual-channel model (will gain EigenSpace PE) |
| `trainer.py`                | Generic training loop |
| `train.py`                  | Training entry point |
| `configurator.py`           | Config override mechanism |
| `generate.py`               | Inference / sampling entry point |
| `midi_viz.py`               | MIDI visualization (Plotly) |
| `play_mpe.py`               | Playback of MPE MIDI |
| `utils.py`, `mingpt_utils.py` | Shared helpers |

### Obsolete — flag, clean later
| File | Notes |
|------|-------|
| `build_parallel_dataset.py` | Superseded by `generate_53tet_dataset.py` |
| `04_runner.py`              | Old training runner |
| `debug_run.py`              | Ad-hoc debug script |
| `test_metadata_export.py`   | One-off metadata dump |
| `test_something_standalone.py` | Scratch |
| `to_compare.py`             | Scratch |
| `utils_lenghts.py`          | Dead constants, folded into `formats.py` where needed |

### Drifted dev notebooks — flag, clean later
`01_musicXML_parser.ipynb`, `02_EigenSpace_mapping.ipynb`,
`03_map_MIDI_to_EigenSpace.ipynb`, `03_map_MIDI_to_FTT.ipynb`,
`04_53TET_conversion.ipynb`, `04_test_mpe.ipynb`,
`04.5_data_augmentation_53edo.ipynb`, `05_generation.ipynb`,
`05_mpe_tokenizer.ipynb`, `05_play_mpe53.ipynb`,
`05.5_tokenization_quality_check.ipynb`, `06_plots_and_figures.ipynb`,
`10_generate_midi_v2_fresh.ipynb`, `11_dataset_qc.ipynb`,
`MIDI_test.ipynb`, `run.ipynb`, `testing_something.ipynb`.

These contain historical exploration. Treat as read-only until cleanup phase.

---

## 10. Requirements checklist

Data / labels:
- [x] Type of transformation tokenized (from folder name) — `TYPE_*`
- [x] Style token present — 16 canonical buckets from `formats.py`
- [x] Chord root tokenized (letter in L1; 53-TET realization in L2)
- [x] Time sequence preserved (duration per chord + bar markers)
- [x] Bar token — plain `BAR`, no numbering
- [x] Style correctly named (e.g. `Jazz`, not `Swing`) via `formats.correctStyleTokensInMeta`
- [x] Style at the right level of information (song-level header)
- [x] Form token emitted (`Form_A`, `Form_B`, …)

Tokenization correctness:
- [x] Chord data correctly encoded (symbol in L1, voicing in L2)
- [x] Chord duration encoded (`DUR_*` in L2, explicit float in L1)
- [x] Bars encoded
- [x] Conditioning context at generation: type + style + first chord

Engineering:
- [x] Remove stray `ROOT_<pc>` tokens from `tokenizer.py` (use L1 letter)
- [x] Implement `parse_text_sidecar()` for L1
- [x] Implement `merge_levels()` to interleave L1+L2 by chord index
- [x] Compute `.eigen.npy` sidecars per file (672 518 / 672 840 = 99.95 %)
- [x] Add EigenSpace MLP to `model.py` and sum into positional embedding
- [x] Pack v3 — `pack_data_v3.py` with Visibility Rule B spans (2.71 B train tokens, 660 870 seqs)
- [ ] Training run `modelA_hybrid_v1`
- [ ] Evaluation: form adherence, voicing plausibility, style match
- [ ] Begin Model B (53-TET holdrian column L2)

---

## 11. Roadmap

**Phase A — Hybrid symbolic + MIDI/MPE (current)**
1. Tokenizer completion (L1 parser, L2 reuse, interleaver)
2. EigenSpace sidecar + positional embedding
3. Pack + train Model A
4. Evaluate form, style, voicing quality

**Phase B — Hybrid symbolic + 53-TET column**
1. Define column L2 vocabulary
2. Train Model B on same L1 with new L2
3. Compare A vs B on identical prompts

**Phase C — Cleanup & paper**
1. Retire obsolete files and drifted notebooks
2. Freeze final tokenizer, model, checkpoints
3. Paper = this document, expanded with results

---

## 12. Archived design

The earlier chord-as-column draft is preserved at
`STRATEGY_V3_original_column_draft.md`. It informs Phase B and should not be
edited as part of Phase A work.

---

## 13. Reproduce / resume

A fresh session (or a fresh machine) can pick up work by running these
stages in order. Each stage is idempotent; re-running it simply overwrites.

### 13.1 Environment

- conda env: **`anima`** — python at
  `/home/david/miniconda3/envs/anima/bin/python`.
- Run all scripts from `src/` (imports assume the src dir is on `sys.path`;
  the scripts add it automatically).

### 13.2 Stage map

| # | Stage                     | Script                               | Output                                                                                       | Typical time |
|---|---------------------------|--------------------------------------|----------------------------------------------------------------------------------------------|--------------|
| A | Build paired dataset      | `src/generate_53tet_dataset.py --workers 8` | `dataset/midi_files/53_tet_mpe/**/*.mid` + `dataset/text_files/53_tet_mpe/**/*.txt`          | hours        |
| B | Build EigenSpace sidecars | `src/build_eigen_sidecars.py --workers 12` | `<stem>.eigen.npy` next to every `.mid` (99.95 % coverage; 322 skipped)                      | ~7 min       |
| C | Pack tokens + eigen       | `src/pack_data_v3.py --workers 24`   | `dataset/tokenized/{train,val}_{tokens,eigen}.bin`, `meta.json`, `vocab.json`                | ~7 min       |
| D | Train Model A             | `src/train.py` (run: `modelA_hybrid_v1`) | `checkpoints/modelA_hybrid_v1/{latest,best,final}.pt`                                        | hours–days   |

Canonical launch command for stage D (Large / GPT-2-small preset, chosen
2026-04-17; see §9 task 7 for rationale):

```bash
cd src/
python train.py \
    --n-layer 12 --n-head 12 --n-embd 768 \
    --batch-size 8 --grad-accum 8 \
    --max-iters 50000 \
    --warmup-iters 2000 \
    --eval-interval 500 \
    --save-interval 2000 \
    --checkpoint-dir ../checkpoints/modelA_hybrid_v1 \
    --wandb-run-name modelA_hybrid_v1
```

Note: `--lr-decay-iters` defaults to `--max-iters` when omitted, so the
cosine schedule automatically contracts to match the shorter run.

If the 32 GB RTX 5090 OOMs at these shapes, halve the live batch and
double accumulation — `--batch-size 4 --grad-accum 16` keeps the effective
batch identical (64) at half the activation memory.

The 322 songs that fail the L1/L2 chord-count invariant in stage B are
**auto-skipped** at stage C (no `.eigen.npy` → not packed). This is the
locked policy for Phase A.

### 13.3 Packed-bin data contract (stage C output)

In `dataset/tokenized/`:

- `train_tokens.bin` — `np.uint16`, shape `(N_train_seqs, block_size + 1)`.
- `train_eigen.bin`  — `np.float16`, shape `(N_train_seqs, block_size + 1, 4)`.
- `val_tokens.bin` / `val_eigen.bin` — same layout, validation split.
- Pad id is **0**; padded positions carry `HEADER_EIGEN = [1.0, 1.0, 2.0, 0.0]`.
- `meta.json` — `vocab_size`, `block_size`, `seq_len`, song and sequence
  counts, per-dim eigenspace (mean, std) over the **training set**, and the
  exact split seed.
- `vocab.json` — authoritative token ↔ id mapping (written by
  `MPETokenizer.save_vocab`); always load it before training / generation.

Training loads the four `.bin` files with `np.memmap`, slices `[:-1]` for
inputs and `[1:]` for targets, and feeds eigen as `float32` into
`GPT2.forward(..., eigen=eigen)`.

### 13.4 Known artifacts to trust

- `src/l1.py` — `parse_text_sidecar`, `merge_levels`, `build_merged_tokens`,
  `merge_levels_with_spans`, `build_merged_tokens_with_spans`.
- `src/eigen_sidecar.py` — per-song `(α, β, γ, D)` builder.
- `src/eigenspace.py` — `classify_intervals` returns a 3-tuple (root-invariant);
  `DissonanceMap.lookup` returns `D`.
- `src/tokenizer.py` — vocab 2930, no `ROOT_*` tokens.
- `src/model.py` — `ModelConfig(vocab_size=2930, block_size=4096, n_eigen=4)`,
  `EigenSpacePositionalEncoding`.
- `src/pack_data_v3.py` — the only supported packer. Old `preprocess.py` /
  `pack_data.py` are retired (kept for reference until Phase C cleanup).

### 13.5 If a stage is interrupted

Stages B and C both re-scan the filesystem on every run and overwrite;
safe to rerun from scratch. Stage D resumes from
`checkpoints/modelA_hybrid_v1/latest.pt` if present.
