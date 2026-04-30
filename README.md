[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Microtonal GPT for 53-TET Chord Progressions

A GPT-2 model that generates **harmonic chord progressions in 53-TET microtonality with learned voicings**, conditioned on transformation type, style, form, tonality, and time structure. Trained on a 672,840-song parallel dataset of jazz-derived progressions transposed and augmented into 53-EDO.

The repository contains the full pipeline from raw chord charts to a trained transformer that emits MPE-MIDI in 53-TET, plus a second model variant that generates a pure microtonal symbolic stream and translates to MIDI deterministically after sampling.

## Why 53-TET

31-EDO already extends 12-TET with five qualities of third (and of every interval class): **subminor, minor, neutral, major, supermajor**. That alone opens a far richer harmonic palette than 12-TET, where minor and major exist.

53-EDO keeps those same five qualities but adds a finer inflection: each one comes in an **up** and **down** variant (notated `^` and `v` in this project's L1 vocabulary). That gives ten shades per interval class instead of five, which is what we actually need to spell common-practice consonances *and* the comma-separated alternatives that distinguish, for example, a 5-limit major third from a Pythagorean one.

So 53-EDO is the smallest tuning that gives us:

1. Same chord colors of 12-TET.
2. The qualitative vocabulary of 31-EDO (sub / minor / neutral / major / super), and
3. A second axis (up / down) that resolves the commas 31-EDO still glosses over,


## Dataset

Parallel twin-tree, 1:1 mirror between MIDI and text sidecars:

```
dataset/midi_files/53_tet_mpe/<type_dir>/<stem>.mid
dataset/text_files/53_tet_mpe/<type_dir>/<stem>.txt
```

- **672,840 paired files**, built from ~4,000 iReal Pro jazz standards expanded through voicing, 12-key transposition, and 14 microtonal transformations.
- **14 transformation types** (`type_0_major`, `type_0_minor`, … `type_6_neutral_n`) — each is a different mapping from 12-TET sonorities into 53-EDO regions.
- **MIDI** carries the voicing as MPE (per-channel pitch-bend) for accurate microtonal playback.
- **Text sidecar** carries everything MIDI cannot: style (16 canonical buckets), tonality (key), form markers (`Form_A/B/C/D`, intro, head, coda, segno), repeat barlines (`|:` `:|`), and per-chord `. <duration> <root> <quality> <extensions> [/ <bass>]`.

Sidecar example:
```
<style> Latin Form_A |: . 4.0 C maj7 | . 4.0 A m7 | ... :|
```

Builder: [src/generate_53tet_dataset.py](src/generate_53tet_dataset.py).

## Approach: hybrid L1 + L2 stream

Per chord, the model sees two aligned blocks in a single autoregressive stream:

```
[ L1 symbolic tokens ]   [ L2 voicing tokens ]
```

- **L1** is the semantic label — what chord, in what form, in what style.
- **L2** is the realization — which 53-TET pitches at which durations and velocities.
- Alignment is by chord index: the k-th L1 chord block matches the k-th L2 `CHORD_START…CHORD_END` block.
- Song-level tokens (`<style>`, `<tonality>`, `TYPE_*`) appear once at the start; structural markers (`|`, `|:`, `:|`, `Form_*`) sit between chord blocks at their bar boundaries.
- One vocabulary, one softmax head, one loss. No dual decoder.

Worked example:

```
<start>
STYLE_Jazz TONALITY_C_major TYPE_0_major FORM_A |:
. 4.0 C maj7                                                          ← L1 block
CHORD_START DUR_4.0 P_212 V_3 P_243 V_3 P_265 V_2 P_284 V_3 CHORD_END  ← L2 block
BAR
. 4.0 A m7
CHORD_START DUR_4.0 P_209 V_3 P_240 V_3 P_262 V_2 P_281 V_3 CHORD_END
:|
<end>
```

## Two models, one comparison

| Aspect    | **Model A** — Hybrid Symbolic + MIDI/MPE     | **Model B** — Hybrid Symbolic + 53-TET Column      |
|-----------|-----------------------------------------------|----------------------------------------------------|
| L1        | Symbolic text sidecar                         | Same                                               |
| L2        | MIDI/MPE pitch + velocity tokens (`P_<step>`) | Pure 53-EDO holdrian tokens (`H_<step>_<vel>`)     |
| Output    | MIDI directly from generation                 | Symbolic stream → deterministic MIDI translator    |
| Purpose   | Learn voicings from real MIDI realizations    | Learn pure microtonal spelling, MIDI-free training |

Both share `block_size = 4096`, `vocab_size ≈ 2930`, GPT-2-small architecture (L12, H12, E768, ~90M params), and the same EigenSpace positional input.

## EigenSpace as positional embedding

Each chord has a 4-vector `(α, β, γ, D)` from a geometric chord-space construction (see [src/eigenspace.py](src/eigenspace.py)). Three coordinates encode interval-class structure; D is a dissonance measure. The vector is **not tokenized** — it is projected through a small MLP (4 → n_embd) and added to the positional embedding before the transformer blocks.

Visibility rule: header tokens use a fixed default; chord blocks carry their own vector; structural markers between chord *k–1* and chord *k* inherit *k–1*'s vector (no future leakage). End-of-sequence inherits the last chord; padding uses the header default.

This injects harmonic geometry as a continuous bias while leaving the discrete vocabulary clean.

## Repository layout

```
src/
  generate_53tet_dataset.py      Builder of the parallel twin-tree dataset
  build_eigen_sidecars.py        Parallel CLI for per-song (α, β, γ, D) sidecars
  eigen_sidecar.py, eigenspace.py
  formats.py, convention.py      Canonical style / note / quality definitions
  chord_mapping.py               53-TET note names and chord interval maps
  voicing.py                     Chord symbol → voicing realization
  transposition.py               Key/root transposition helpers
  l1.py                          L1 parser + L1+L2 interleaver (with eigen spans)
  tokenizer.py                   Method A tokenizer (P_<step> + V_<v>)
  tokenizer_b.py                 Method B tokenizer (H_<step>_<vel>)
  pack_data_v3.py                Method A packer  → dataset/tokenized/
  pack_data_b.py                 Method B packer  → dataset/tokenized_b/
  model.py                       GPT-2 with EigenSpace positional MLP
  train.py, trainer.py, configurator.py
  generate.py, generate_b.py     Inference entries
  holdrian_to_midi.py            Method B post-hoc 53-EDO → MPE-MIDI translator
  midi_viz.py, play_mpe.py
  paper.tex                      Paper source
  listening_test.js              Browser AB-test frontend
  01_..12_*.ipynb                Pipeline + analysis notebooks (in order)
dataset/
  midi_files/53_tet_mpe/<type_dir>/<stem>.mid
  text_files/53_tet_mpe/<type_dir>/<stem>.txt
  tokenized/                     Method A packed bins + meta.json + vocab.json
  tokenized_b/                   Method B packed bins
checkpoints/
  modelA_hybrid_v1/              Method A weights
  modelB_column_v1/              Method B weights
```

## Vocabulary at a glance

**Method A — total vocab 2,930**

| Group         | Count | Notes                                         |
|---------------|------:|-----------------------------------------------|
| Special       |     4 | `<pad> <start> <end> <sep>`                   |
| Structural    |     4 | `CHORD_START`, `CHORD_END`, `BAR`, `REST`     |
| Duration      |    10 | `DUR_*` ∈ {0.5, 1.0, …, 16.0}                 |
| Type / Style / Form | 14 + 16 + 9 | Conditioning labels                  |
| Pitch (L2)    |   319 | `P_106 … P_424`, absolute 53-EDO step         |
| Velocity (L2) |     8 | `V_1 … V_8`                                   |
| L1 symbolic   |   ~321| Roots (71), qualities (193), extensions (16), tonality, structurals |

**Method B** replaces the 319 `P_*` × 8 `V_*` tokens with a flat compound `H_<step>_<vel>` vocabulary (319 × 8 = 2,552 tokens). The total vocab size is held at ~2,930 so the model architecture loads identically and the A↔B comparison isolates one variable: the *interpretation* of the pitch token (MIDI-derived vs. pure 53-EDO).

## Reproducing

Stages are idempotent; rerun any stage by simply running the script.

```bash
# A. Build the paired 53-TET dataset (hours)
python src/generate_53tet_dataset.py --workers 8

# B. Build EigenSpace sidecars (~7 min, 672,518 / 672,840 = 99.95% coverage)
python src/build_eigen_sidecars.py --workers 12

# C. Pack tokens + eigen for Model A (~7 min)
python src/pack_data_v3.py --workers 24
# → dataset/tokenized/{train,val}_{tokens,eigen}.bin + meta.json + vocab.json

# D. Train Model A (GPT-2-small preset, ~90M params)
python src/train.py \
    --n-layer 12 --n-head 12 --n-embd 768 \
    --batch-size 8 --grad-accum 8 \
    --max-iters 50000 --warmup-iters 2000 \
    --checkpoint-dir checkpoints/modelA_hybrid_v1

# E. Generate
python src/generate.py     # Model A: writes MPE-MIDI directly
python src/generate_b.py   # Model B: writes 53-EDO stream, then holdrian_to_midi
```

If your GPU OOMs, halve `--batch-size` and double `--grad-accum` to keep the effective batch (64) constant.

## Evaluation

Models are compared on identical prompts using:

1. **Form adherence** — balanced `|:`/`:|` and detected section change after `Form_A`.
2. **Symbol↔voicing consistency** — L2 pitch set maps back to the emitted L1 symbol within 53-TET tolerance.
3. **Style and type conditioning** — fixing one and varying the other shifts L1 / L2 statistics measurably.
4. **Novelty vs. memorization** — n-gram overlap with the training set on both L1 and L2.
5. **Listening test** — browser-based AB test (see [src/listening_test.js](src/listening_test.js) and [src/12_listening_test_dataset.ipynb](src/12_listening_test_dataset.ipynb)).

## Requirements

Python 3.10+, PyTorch, `pretty_midi`, `mido`, `music21`, `numpy`. The trainer follows the minGPT layout ([src/model.py](src/model.py), [src/configurator.py](src/configurator.py), [src/trainer.py](src/trainer.py)).

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). See the bundled license file or [creativecommons.org/licenses/by-nc/4.0](https://creativecommons.org/licenses/by-nc/4.0/).
