# ANIMA Microtonal GPT — Strategy V3

Single source of truth for the current phase.
Contains: the goal, the data contract, the tokenization spec, the model-input
plan, the log of decisions and past failures, and the road to follow.
This document also serves as the paper outline.

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

Canonical source: `src/formats.py`. Reuse `correctStyleTokensInMeta`,
`getNotes`, `getNatures`, `listToIgnore`, `splitChordTokens`,
`splitSlashChords`. Do not reinvent these tables.

| Group        | Count | Examples                                                    | Source |
|--------------|------:|-------------------------------------------------------------|--------|
| Header       | 3     | `<style>`, `<tonality>`, `<type>`                           | literal |
| Type         | 14    | `TYPE_0_major` … `TYPE_6_neutral_n`                         | folder name |
| Style        | 16    | `Jazz Blues Folk Bossa Reggae Samba Funk Pop Son Rock Soul Balad RnB Gospel Afoxé "Even 8ths"` | `formats.correctStyleTokensInMeta` |
| Tonality     | ~24   | `C_major`, `A_minor`, …                                     | sidecar |
| Form         | 9     | `FORM_INTRO A B C D VERSE HEAD CODA SEGNO`                  | sidecar |
| Structural   | 4     | `|`, `|:`, `:|`, `BAR`                                      | sidecar |
| Chord start  | 1     | `.`                                                         | sidecar |
| Duration (L1)| open  | float literal beats (`4.0`, `2.0`, `0.5`, …)                | sidecar |
| Root letter  | 20    | from `getNotes()`                                           | sidecar |
| Quality      | 18    | from `getNatures()` (`maj, maj7, m, m7, dom7, ø7, o7, sus, aug, …`) | sidecar |
| Extensions   | open  | `b9`, `#11`, `add6`, …                                      | sidecar |
| Slash bass   | 1+20  | `/` + root letter                                           | sidecar |

Rules:
- Style is chosen by the substring rules in `formats.correctStyleTokensInMeta`. **Diversity preserved (16 buckets, not collapsed to 4).**
- Root in L1 is a **letter**, not a pitch class. 53-TET realization lives in L2.
- `BAR` is a plain structural marker. **No numbering.**

---

## 5. Level 2 vocabulary (MIDI/MPE, from `.mid`)

Extracted from `src/tokenizer.py` (current state, vocab_size = 2662):

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

Known stray to remove: `ROOT_<pc>` tokens. Root in L1 is the chord letter;
L2 carries the 53-TET voicing. `ROOT_*` in the vocab is a leftover of an
earlier misconception and must be deleted (see §9 Open tasks).

Implementation: `MPETokenizer.encode_chords` / `chords_to_midi` in
`src/tokenizer.py`.

---

## 6. EigenSpace — positional embedding, not a token

Each chord has a 4-vector `(α, β, γ, D)` from `src/eigenspace.py`.
It is **not tokenized**.

Pipeline:
1. Compute per-chord `(α, β, γ, D)` → save as `<stem>.eigen.npy`.
2. At model input, broadcast the chord vector across every token of its
   L1+L2 block.
3. Project `(4 → n_embd)` via a small MLP and **add** to the positional
   embedding before the transformer.

**Open decision**: whether to broadcast across the whole block or inject
only at `CHORD_START`. Default chosen for Phase A: broadcast across the
block. Revisit if form adherence is poor.

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

---

## 9. Open tasks (the road)

### Phase A — Hybrid Symbolic + MIDI/MPE (current)

1. **Tokenizer cleanup**
   - Remove `ROOT_<pc>` tokens and all code paths that emit them
   - Confirm vocab size after removal; save `vocab.json`
2. **L1 parser** — `parse_text_sidecar(path) -> list[Token]`
   - Reads a `.txt`, yields header tokens once, then per-chord L1 blocks
     interleaved with structural markers
3. **L1+L2 interleaver** — `merge_levels(txt_tokens, mid_tokens) -> list[int]`
   - Aligns by chord index; emits
     `<start> …header… [L1_block_k, L2_block_k for k in chords]… <end>`
   - Asserts equal chord counts on both sides; logs and skips on mismatch
4. **EigenSpace sidecar builder**
   - For each `<stem>.mid`, compute chord sequence + `(α, β, γ, D)` per chord,
     save `<stem>.eigen.npy` next to the MIDI
5. **Model update** (`src/model.py`)
   - Add `nn.Linear(4, n_embd)` (or small MLP) for EigenSpace
   - At forward: broadcast chord vector across each chord's token span,
     project, and add to positional embedding
6. **Preprocess + pack** (`preprocess.py`, `pack_data.py`)
   - Write the interleaved stream + aligned EigenSpace tensor to packed bins
7. **Train Model A** (`train.py`)
   - Run name: `modelA_hybrid_v1`
   - Save to `checkpoints/modelA_hybrid_v1/`
8. **Evaluate Model A** (§10)

### Phase B — Hybrid Symbolic + 53-TET column

Only start after A is evaluated. Reuse L1. Replace L2 with a full 53-TET
holdrian-column representation (design TBD, to be specified here when
Phase A is complete). Translator to MIDI runs after generation.

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
- [ ] Remove stray `ROOT_<pc>` tokens from `tokenizer.py` (use L1 letter)
- [ ] Implement `parse_text_sidecar()` for L1
- [ ] Implement `merge_levels()` to interleave L1+L2 by chord index
- [ ] Compute `.eigen.npy` sidecars per file
- [ ] Add EigenSpace MLP to `model.py` and sum into positional embedding
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
