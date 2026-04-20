"""
tokenizer_b.py
==============
Method B tokenizer — **pure holdrian-comma** L2 vocabulary for the
symbolic 53-TET GPT described in STRATEGY_V3.md §B.1 (locked 2026-04-20).

Method B vs Method A
--------------------
- Method A (``tokenizer.py``) emits compound MIDI-derived pitch+velocity
  tokens ``PV_<step>_<vel>`` and renders MIDI directly from the generated
  stream.
- Method B (this file) emits compound **holdrian comma** + velocity
  tokens ``H_<step>_<vel>``. The vocabulary has **zero MIDI semantics**.
  MIDI only appears at inference time via the deterministic translator
  ``src/holdrian_to_midi.py`` (task B-4).

The block layout, sequence protocol, DUR grid, velocity binning, max
notes per chord, L1 alphabet, conditioning tokens (TYPE_/STYLE_/FORM_)
and special/structural tokens are **identical** to Method A so that
``model.py`` can be imported and used unchanged (isolation contract §B.0
rules 1, 2, 6).

This module:
  * Imports ``tokenizer.py`` read-only for constants, helper functions,
    and the raw MIDI parser (``parse_mpe_midi`` / ``clean_chords``).
  * Defines a **new** class ``TokenizerB`` — it is **not** a subclass of
    ``MPETokenizer``.
  * Writes its own ``vocab.json`` under ``dataset/tokenized_b/``.

Per-chord token block (Method B)::

    CHORD_START  DUR_<d>  H_<step>_<vel>  H_<step>_<vel>  ...  CHORD_END

  * ``step`` ∈ [106, 424] — absolute 53-EDO step (319 values).
  * ``vel``  ∈ [1, 8]     — velocity bin (8 values).
  * Notes inside a chord are ordered low→high by ``step``.
  * ``MAX_CHORD_NOTES = 8``.
"""

from __future__ import annotations

import json
from pathlib import Path

# Read-only import of Method A helpers. Isolation contract §B.0 rule 2
# permits ``import`` of frozen modules; we only **consume** symbols here.
from tokenizer import (
    PAD_TOKEN, START_TOKEN, END_TOKEN, SEP_TOKEN,
    CHORD_START_TOKEN, CHORD_END_TOKEN, BAR_TOKEN, REST_TOKEN,
    DURATION_GRID, NUM_VELOCITY_BINS, MAX_CHORD_NOTES,
    PITCH_OFFSET, MAX_53TET_STEP,
    TYPE_PREFIX, TYPE_LABELS,
    STYLE_PREFIX, STYLE_LABELS,
    FORM_PREFIX, FORM_TOKENS,
    classify_form,
    quantize_duration, quantize_velocity, dequantize_velocity,
    parse_mpe_midi, clean_chords,
    _extract_type_label,
)


# =============================================================================
# Method-B pitch-token prefix
# =============================================================================

# Method A uses ``PV_<step>_<vel>``. Method B uses ``H_<step>_<vel>``.
# Same integer grid, different semantic interpretation (holdrian comma,
# not MIDI note). See STRATEGY_V3.md §B.1.
PITCH_PREFIX_B = "H"

# Method B inherits Method A's pitch range exactly so generation statistics
# are comparable note-for-note across the two models.
PITCH_OFFSET_B = PITCH_OFFSET        # 106
MAX_53TET_STEP_B = MAX_53TET_STEP    # 424


# =============================================================================
# TokenizerB
# =============================================================================

class TokenizerB:
    """
    Method B tokenizer — holdrian-comma L2 vocabulary.

    Vocabulary order (deterministic; mirrors Method A for apples-to-apples
    comparison):

      1. Special: ``<pad> <start> <end> <sep>``
      2. Structural: ``CHORD_START CHORD_END BAR REST``
      3. Duration: ``DUR_*`` (same grid as A)
      4. Type: ``TYPE_*`` (14)
      5. Style: ``STYLE_*`` (16)
      6. Form: ``FORM_*`` (9)
      7. **Holdrian+velocity**: ``H_<step>_<vel>`` for
         ``step ∈ [PITCH_OFFSET_B, MAX_53TET_STEP_B]`` × ``vel ∈ [1, 8]``
      8. L1 symbolic alphabet via :func:`l1.load_alphabet`.

    Attributes:
        token_to_id (dict[str, int])
        id_to_token (dict[int, str])
        vocab_size (int)
    """

    def __init__(
        self,
        max_pitch: int = MAX_53TET_STEP_B,
        num_vel_bins: int = NUM_VELOCITY_BINS,
        duration_grid=None,
        beats_per_bar: int = 4,
        pitch_offset: int = PITCH_OFFSET_B,
    ):
        self.max_pitch = max_pitch
        self.num_vel_bins = num_vel_bins
        self.duration_grid = list(duration_grid) if duration_grid is not None else list(DURATION_GRID)
        self.beats_per_bar = beats_per_bar
        self.pitch_offset = pitch_offset

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._l1_alphabet = None
        self._warned_tokens: set[str] = set()

        self._build_vocab()

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def _build_vocab(self) -> None:
        tokens: list[str] = []

        # 1. Special
        tokens.extend([PAD_TOKEN, START_TOKEN, END_TOKEN, SEP_TOKEN])

        # 2. Structural
        tokens.extend([CHORD_START_TOKEN, CHORD_END_TOKEN, BAR_TOKEN, REST_TOKEN])

        # 3. Duration
        for dur in self.duration_grid:
            tokens.append(f"DUR_{dur}")

        # 4. Type conditioning
        for label in TYPE_LABELS:
            tokens.append(f"{TYPE_PREFIX}_{label}")

        # 5. Style conditioning
        for label in STYLE_LABELS:
            tokens.append(f"{STYLE_PREFIX}_{label}")

        # 6. Form markers
        tokens.extend(FORM_TOKENS)

        # 7. Holdrian compound pitch+velocity tokens (Method B replacement
        #    for Method A's PV_ family). Same integer grid, different name.
        for step in range(self.pitch_offset, self.max_pitch + 1):
            for v in range(1, self.num_vel_bins + 1):
                tokens.append(f"{PITCH_PREFIX_B}_{step}_{v}")

        # 8. L1 symbolic alphabet (reused unchanged from Method A pipeline).
        try:
            from l1 import load_alphabet  # type: ignore
            self._l1_alphabet = load_alphabet()
            tokens.extend(self._l1_alphabet.vocab_tokens())
        except Exception as exc:
            print(f"  [tokenizer_b] L1 alphabet not loaded: {exc}")
            self._l1_alphabet = None

        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(tokens)}
        self.vocab_size = len(tokens)

    # ------------------------------------------------------------------
    # Encoding: chord events → tokens
    # ------------------------------------------------------------------

    def encode_chords(
        self,
        chords,
        add_start_end: bool = True,
        type_label: str | None = None,
        style_label: str | None = None,
        form_markers: dict | None = None,
    ) -> list[str]:
        """
        Encode a list of chord event dicts into the Method-B token stream.

        Mirrors :meth:`tokenizer.MPETokenizer.encode_chords` exactly in
        protocol (DUR = onset-to-onset delta, last chord uses its own
        note-off duration, bar boundaries emit ``BAR`` with optional
        leading ``FORM_<X>``, notes sorted low→high, capped at
        ``MAX_CHORD_NOTES``). The only difference is the pitch token
        prefix: ``H_`` instead of ``PV_``.
        """
        tokens: list[str] = []

        if add_start_end:
            tokens.append(START_TOKEN)

        if type_label is not None:
            type_token = f"{TYPE_PREFIX}_{type_label}"
            if type_token in self.token_to_id:
                tokens.append(type_token)
            else:
                print(f"  Warning: unknown type label '{type_label}', skipping TYPE token")

        if style_label is not None:
            style_token = f"{STYLE_PREFIX}_{style_label}"
            if style_token in self.token_to_id:
                tokens.append(style_token)
            else:
                print(f"  Warning: unknown style label '{style_label}', skipping STYLE token")

        # Normalise form_markers: {bar_0idx:int → FORM_<LABEL>:str}
        form_lookup: dict[int, str] = {}
        if form_markers:
            for bar_idx, raw in form_markers.items():
                canon = classify_form(raw)
                if canon is None:
                    continue
                ftok = f"{FORM_PREFIX}_{canon}"
                if ftok in self.token_to_id:
                    form_lookup[int(bar_idx)] = ftok

        if 0 in form_lookup:
            tokens.append(form_lookup[0])

        last_bar = -1
        for i, chord in enumerate(chords):
            current_bar = int(chord['onset_beats'] // self.beats_per_bar)
            if current_bar > last_bar:
                bars_to_emit = current_bar - max(0, last_bar)
                for b in range(bars_to_emit):
                    if last_bar >= 0:
                        bar_0idx = last_bar + b + 1
                        if bar_0idx in form_lookup:
                            tokens.append(form_lookup[bar_0idx])
                        tokens.append(BAR_TOKEN)
                last_bar = current_bar

            tokens.append(CHORD_START_TOKEN)

            if i < len(chords) - 1:
                delta = chords[i + 1]['onset_beats'] - chord['onset_beats']
                q_dur = quantize_duration(max(0.5, delta))
            else:
                q_dur = quantize_duration(chord['duration_beats'])
            tokens.append(f"DUR_{q_dur}")

            sorted_notes = sorted(chord['notes'], key=lambda n: n['step_53'])[:MAX_CHORD_NOTES]
            for note in sorted_notes:
                step = max(self.pitch_offset, min(self.max_pitch, note['step_53']))
                vel_bin = quantize_velocity(note['velocity'], self.num_vel_bins)
                tokens.append(f"{PITCH_PREFIX_B}_{step}_{vel_bin}")

            tokens.append(CHORD_END_TOKEN)

        if add_start_end:
            tokens.append(END_TOKEN)

        return tokens

    def encode_file(
        self,
        midi_path,
        speed: float = 1.0,
        add_start_end: bool = True,
        type_label: str | None = None,
        style_label: str | None = None,
        form_markers: dict | None = None,
    ) -> list[str]:
        """
        Parse an MPE MIDI file and encode it as a Method-B token stream.

        The .mid file is only used as the *source of chord events* (step_53
        integers + velocity). No MIDI semantics leak into the vocabulary.
        """
        chords = parse_mpe_midi(midi_path, speed=speed)
        if not chords:
            return []
        chords = clean_chords(chords)
        if type_label is None:
            type_label = _extract_type_label(midi_path)
        return self.encode_chords(
            chords,
            add_start_end=add_start_end,
            type_label=type_label,
            style_label=style_label,
            form_markers=form_markers,
        )

    # ------------------------------------------------------------------
    # Token ↔ id
    # ------------------------------------------------------------------

    def encode_to_ids(self, tokens) -> list[int]:
        pad_id = self.token_to_id[PAD_TOKEN]
        ids: list[int] = []
        for t in tokens:
            tok_id = self.token_to_id.get(t)
            if tok_id is None:
                if t not in self._warned_tokens:
                    self._warned_tokens.add(t)
                    print(f"  Warning: token '{t}' not in vocabulary, mapping to <pad>")
                tok_id = pad_id
            ids.append(tok_id)
        return ids

    def decode_ids(self, ids) -> list[str]:
        return [self.id_to_token.get(int(i), PAD_TOKEN) for i in ids]

    # ------------------------------------------------------------------
    # Decoding: tokens → chord events (53-EDO integers, no MIDI)
    # ------------------------------------------------------------------

    def decode(self, tokens) -> list[dict]:
        """
        Decode a Method-B token sequence back into chord events.

        Returns chord dicts with the same shape used by Method A::

            {'onset_beats': float,
             'duration_beats': float,
             'notes': [{'step_53': int, 'velocity': int}, ...]}

        ``step_53`` here is a **holdrian comma index**, not a MIDI note.
        Translation to MIDI (if desired) is the job of
        ``src/holdrian_to_midi.py`` (task B-4).
        """
        chords: list[dict] = []
        current_beat = 0.0
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok == BAR_TOKEN or (isinstance(tok, str) and tok.startswith('BAR_')):
                i += 1

            elif tok == CHORD_START_TOKEN:
                i += 1
                duration = 4.0
                notes: list[dict] = []

                while i < len(tokens) and tokens[i] != CHORD_END_TOKEN:
                    t = tokens[i]

                    if isinstance(t, str) and t.startswith("DUR_"):
                        try:
                            duration = float(t[4:])
                        except ValueError:
                            pass

                    elif isinstance(t, str) and t.startswith(f"{PITCH_PREFIX_B}_"):
                        # H_<step>_<vel>
                        parts = t.split("_")
                        if len(parts) == 3:
                            try:
                                step_53 = int(parts[1])
                                vel_bin = int(parts[2])
                                vel = dequantize_velocity(vel_bin, self.num_vel_bins)
                                notes.append({'step_53': step_53, 'velocity': vel})
                            except ValueError:
                                pass

                    # Unknown tokens (including any Method-A PV_* that might
                    # leak into a mixed corpus) are skipped silently.
                    i += 1

                if notes:
                    chords.append({
                        'onset_beats': round(current_beat, 4),
                        'duration_beats': duration,
                        'notes': notes,
                    })
                    current_beat += duration

                i += 1  # skip CHORD_END_TOKEN
            else:
                i += 1

        return chords

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def pad_sequence(self, token_ids, max_length: int) -> list[int]:
        pad_id = self.token_to_id[PAD_TOKEN]
        if len(token_ids) >= max_length:
            return list(token_ids[:max_length])
        return list(token_ids) + [pad_id] * (max_length - len(token_ids))

    # ------------------------------------------------------------------
    # Vocabulary I/O
    # ------------------------------------------------------------------

    def save_vocab(self, path) -> None:
        data = {
            'method': 'B',
            'pitch_prefix': PITCH_PREFIX_B,
            'token_to_id': self.token_to_id,
            'config': {
                'max_pitch': self.max_pitch,
                'num_vel_bins': self.num_vel_bins,
                'duration_grid': self.duration_grid,
                'beats_per_bar': self.beats_per_bar,
                'pitch_offset': self.pitch_offset,
                'vocab_size': self.vocab_size,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_vocab(cls, path) -> "TokenizerB":
        with open(path, 'r') as f:
            data = json.load(f)
        config = data['config']
        return cls(
            max_pitch=config['max_pitch'],
            num_vel_bins=config['num_vel_bins'],
            duration_grid=config['duration_grid'],
            beats_per_bar=config['beats_per_bar'],
            pitch_offset=config.get('pitch_offset', PITCH_OFFSET_B),
        )
