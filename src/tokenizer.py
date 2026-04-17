"""
tokenizer.py
============
Tokenization module for 53-TET MPE MIDI files.

Converts MIDI Polyphonic Expression (MPE) files into flat token sequences 
suitable for training a GPT-2 model. Each chord event is decomposed into 
a series of sub-tokens representing timing, pitch (in 53-TET steps), and velocity.

Token Vocabulary
----------------
The token vocabulary consists of 5 categories:

  1. SPECIAL TOKENS:
     <pad>, <start>, <end>, <sep>

  2. STRUCTURAL TOKENS:
     CHORD_START, CHORD_END, BAR

  3. DURATION TOKENS:
     DUR_1.0, DUR_1.33, DUR_1.5, DUR_2.0, DUR_2.67, DUR_3.0, DUR_4.0, 
     DUR_6.0, DUR_8.0 ... (quantized to the grid values used in the dataset)

  4. PITCH TOKENS:
     P_<step> where step is the absolute 53-TET pitch (0 to 529).
     Octave 0 = steps 0-52, Octave 1 = 53-105, etc.
     Practical range in the dataset: ~P_130 to P_370.

  5. VELOCITY TOKENS:
     V_<bin> where bin is a quantized velocity level (1-8).

Sequence Format
---------------
A tokenized song looks like:

  <start>  CHORD_START DUR_4.0 P_212 V_3 P_243 V_3 P_265 V_2 P_284 V_3 P_306 V_2 CHORD_END  BAR  CHORD_START DUR_4.0 ...  <end>

The model learns:
  - Which pitches form musically coherent chords (microtonal voicings)
  - How chords progress over time (harmonic rhythm)
  - Stylistic patterns of 53-TET harmony

Usage
-----
  from tokenizer import MPETokenizer

  tokenizer = MPETokenizer()
  tokens = tokenizer.encode_file("path/to/file.mid")
  midi_events = tokenizer.decode(tokens)
  tokenizer.save_vocab("vocab.json")
"""

import json
import mido
import math
import os
import random
from pathlib import Path
from collections import Counter

# Lazy torch import (only needed for dataset __getitem__)
_torch_module = None
def _torch():
    global _torch_module
    if _torch_module is None:
        import torch
        _torch_module = torch
    return _torch_module


# =============================================================================
# CONSTANTS
# =============================================================================

# 53-TET divisions per octave
TET_53 = 53

# Pitch Bend Range (set via RPN in the MIDI files) = ±2 semitones = ±200 cents
PB_RANGE_CENTS = 200.0

# MIDI pitch bend limits
PB_MAX = 8191
PB_MIN = -8192

# Maximum number of notes per chord 
# (Dataset is mostly 5, sometimes 6. Pad to MAX_CHORD_NOTES)
MAX_CHORD_NOTES = 8

# 53-TET pitch range for tokenization
# Across the dataset, actual pitches fall in ~164–336 (steps 3–6 octaves).
# We add a margin of ±1 octave (53 steps) for safety and transposition.
PITCH_OFFSET = 106    # Lowest pitch token = step 106 (octave 2)
MAX_53TET_STEP = 424  # Highest pitch token = step 424 (octave 8)
# Effective range: 319 pitch tokens instead of 531 (40% smaller vocab)

# Velocity quantization bins (1-based: V_1 through V_8)
NUM_VELOCITY_BINS = 8

# Duration quantization grid (in beats)
# Primary grid: actual values found in the dataset (2, 4, 6, 8 beats).
# Extended grid: finer subdivisions for future datasets / generation diversity.
DURATION_GRID = [
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0
]

# Special tokens
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
SEP_TOKEN = "<sep>"

# Structural tokens
CHORD_START_TOKEN = "CHORD_START"
CHORD_END_TOKEN = "CHORD_END"
BAR_TOKEN = "BAR"
REST_TOKEN = "REST"

# Root token: encodes the chord root as a 53-TET pitch class (0–52)
# ROOT_0 = C (approx), ROOT_9 = D (approx), etc.
ROOT_PREFIX = "ROOT"

# Bar number tokens: BAR_1 … BAR_64 replace the anonymous BAR token.
# Positional — the model learns "this is bar 5 of a 12-bar blues", etc.
# 64 covers all practical song lengths (most jazz standards ≤ 32 bars).
BAR_MAX = 64
BAR_NUMBER_TOKENS = [f"BAR_{n}" for n in range(1, BAR_MAX + 1)]

# Type conditioning tokens: encode the 53-TET transformation type
# These are prepended at the START of each sequence (before <start>)
# to give the model explicit control over which tuning system to generate in.
# The 14 types correspond to the folder names in dataset/midi_files/53_tet_mpe/
TYPE_PREFIX = "TYPE"
TYPE_LABELS = [
    "0_major",
    "0_minor",
    "1_minor",
    "1_neutral",
    "2_minor",
    "2_subminor",
    "3_major",
    "3_minor",
    "4_minor",
    "4_upmajor",
    "5_major_v2",
    "5_minor",
    "6_minor",
    "6_neutral_n",
]
TYPE_TOKENS = [f"{TYPE_PREFIX}_{label}" for label in TYPE_LABELS]

# Style conditioning tokens: encode the musical genre/style of the source song
# These are prepended after the TYPE token at the START of each sequence.
# Raw iReal Pro styles (131 unique values) are grouped into 10 canonical categories.
STYLE_PREFIX = "STYLE"
STYLE_LABELS = [
    "jazz",
    "bossa_samba",
    "ballad",
    "pop",
    "rock",
    "waltz",
    "funk_soul",
    "latin",
    "blues",
    "folk_country",
]
STYLE_TOKENS = [f"{STYLE_PREFIX}_{label}" for label in STYLE_LABELS]

# Form conditioning tokens: encode the position inside the song form
# (intro, verse, head, section A/B/C/D, segno, coda). Emitted at bar
# boundaries where the original iRealXML source contains a <rehearsal>,
# <segno>, or <coda> marker. Gives the model an explicit form signal
# so generation follows phrase-level structure, not just local harmony.
FORM_PREFIX = "FORM"
FORM_LABELS = [
    "INTRO",
    "A",
    "B",
    "C",
    "D",
    "VERSE",
    "HEAD",
    "CODA",
    "SEGNO",
]
FORM_TOKENS = [f"{FORM_PREFIX}_{label}" for label in FORM_LABELS]

# Map raw XML form strings (as produced by xmlTranslator) to canonical FORM labels.
# Raw values look like "Form_A", "Form_intro", "Form_Coda", "Form_Segno".
def classify_form(raw_form):
    """
    Map a raw form marker ('Form_A', 'Form_intro', 'Form_Coda', ...)
    to a canonical FORM_<LABEL>. Returns None if unrecognised.
    """
    if not raw_form:
        return None
    s = str(raw_form).strip()
    if s.lower().startswith("form_"):
        s = s[5:]
    s = s.strip().upper()
    # Normalise a few aliases
    if s in ("IN", "INTR"):
        s = "INTRO"
    if s in ("OUTRO",):
        s = "CODA"
    if s in FORM_LABELS:
        return s
    return None


# Mapping from raw iReal Pro style strings to canonical STYLE labels
_RAW_STYLE_TO_GROUP = None

def _build_style_map():
    """Build the raw-style → canonical-group mapping (lazy, cached)."""
    global _RAW_STYLE_TO_GROUP
    if _RAW_STYLE_TO_GROUP is not None:
        return _RAW_STYLE_TO_GROUP

    import re as _re
    _RAW_STYLE_TO_GROUP = {}

    # Keywords-based classification (order matters: first match wins)
    # Mirrors formats.py logic: anything containing "rock" → rock (dominant).
    # Band names (Beatles, Rolling Stones) also → rock.
    _rules = [
        # Rock FIRST — any style containing "rock" is rock (matches formats.py behavior)
        # Also catches band names: Beatles, Rolling Stones, etc.
        (r'rock|reggae|beatles|rolling.?stones', 'rock'),
        # Ballad (after rock, so "Rock Ballad" → rock, but "Pop Ballad" → ballad)
        (r'ballad', 'ballad'),
        # Samba / Bossa (after rock, so "Samba-Rock" → rock)
        (r'samba|bossa|choro|marchinha|maxixe|frevo|forr|bai[aã]o|afox[eé]|afro', 'bossa_samba'),
        # Blues / Shuffle (after rock, so "Blues Rock" → rock)
        (r'blues|shuffle', 'blues'),
        # Waltz (after rock, so "Rock Waltz" → rock)
        (r'waltz', 'waltz'),
        # Jazz / Swing (broad — catches "medium swing", "up tempo swing", etc.)
        (r'swing|jazz|fusion|even.?8|even.?16|moderately|deliberately|medium\s*slow|slowly|128\s*feel|medium\s*up$|up\s*tempo$|dreamlike', 'jazz'),
        # Pop
        (r'pop|disco|electro|musical', 'pop'),
        # Funk / Soul / R&B
        (r'funk|soul|r.?n.?b', 'funk_soul'),
        # Latin (bolero, tango, son, salsa, etc.)
        (r'latin|bolero|tango|son$|salsa|montuno|mambo|cha\s*cha|merengue|calypso|chacarera|cuban', 'latin'),
        # Folk / Country / Worship
        (r'folk|country|hymn|worship|gospel|march$', 'folk_country'),
    ]

    # We'll populate lazily when first called with actual style strings
    _RAW_STYLE_TO_GROUP['__rules__'] = _rules
    return _RAW_STYLE_TO_GROUP


def classify_style(raw_style):
    """
    Map a raw iReal Pro style string to a canonical STYLE label.

    Args:
        raw_style (str): e.g. "Medium Swing", "Bossa Nova", "Rock Pop"

    Returns:
        str: Canonical label from STYLE_LABELS, or None if empty/unknown
    """
    if not raw_style or not raw_style.strip():
        return None

    import re as _re
    style_map = _build_style_map()
    rules = style_map['__rules__']
    sl = raw_style.strip().lower()

    for pattern, group in rules:
        if _re.search(pattern, sl):
            return group

    # Fallback: if nothing matched, use 'jazz' (dominant class)
    return 'jazz'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_type_label(midi_path):
    """
    Extract the transformation type label from a MIDI file path.
    
    Checks the parent folder name (e.g. "type_0_major") or filename suffix.
    Returns the label portion after "type_" (e.g. "0_major"), or None.
    """
    p = Path(midi_path)
    # Try parent folder name: "type_0_major" → "0_major"
    parent = p.parent.name
    if parent.startswith("type_"):
        label = parent[5:]
        if label in TYPE_LABELS:
            return label
    # Try filename: "..._type_0_major.mid" → "0_major"
    stem = p.stem
    for tl in TYPE_LABELS:
        if f"type_{tl}" in stem:
            return tl
    return None


def midi_bend_to_53tet_step(midi_note, pitch_bend_value):
    """
    Convert a MIDI note + pitch bend (MPE) to an absolute 53-TET step.
    
    The MPE encoding uses per-channel pitch bend with ±200 cents range
    to shift each note to its correct 53-TET pitch.
    
    Args:
        midi_note (int): MIDI note number (0-127)
        pitch_bend_value (int): MIDI pitch bend (-8192 to 8191)
    
    Returns:
        int: Absolute 53-TET step number
    """
    # Convert pitch bend to cents
    cents = (pitch_bend_value / PB_MAX) * PB_RANGE_CENTS
    
    # MIDI note in 53-TET (without bend): note * 53/12
    base_step = midi_note * TET_53 / 12.0
    
    # Add the microtonal deviation: cents / (cents_per_53tet_step)
    # 1 octave = 1200 cents = 53 steps → 1 step = 1200/53 ≈ 22.64 cents
    deviation_steps = cents * TET_53 / 1200.0
    
    return round(base_step + deviation_steps)


def quantize_duration(duration_beats):
    """
    Snap a duration (in beats) to the nearest value in the grid.
    
    Args:
        duration_beats (float): Duration in beats
    
    Returns:
        float: Quantized duration from DURATION_GRID
    """
    if duration_beats <= 0:
        return DURATION_GRID[0]
    
    best = DURATION_GRID[0]
    best_dist = abs(duration_beats - best)
    
    for g in DURATION_GRID:
        dist = abs(duration_beats - g)
        if dist < best_dist:
            best = g
            best_dist = dist
    
    return best


def quantize_velocity(velocity, num_bins=NUM_VELOCITY_BINS):
    """
    Quantize MIDI velocity (0-127) into bins (1 to num_bins).
    
    Args:
        velocity (int): MIDI velocity 0-127
        num_bins (int): Number of output bins
    
    Returns:
        int: Velocity bin (1-based)
    """
    if velocity <= 0:
        return 1
    # Map 1-127 → 1-num_bins
    bin_idx = math.ceil(velocity / 127.0 * num_bins)
    return max(1, min(num_bins, bin_idx))


def dequantize_velocity(vel_bin, num_bins=NUM_VELOCITY_BINS):
    """
    Convert a velocity bin back to a MIDI velocity value.
    
    Args:
        vel_bin (int): Velocity bin (1-based)
        num_bins (int): Number of bins
    
    Returns:
        int: MIDI velocity (1-127)
    """
    return max(1, min(127, round(vel_bin / num_bins * 127)))


def step53_to_midi_and_bend(step_53):
    """
    Convert a 53-TET step back to MIDI note + pitch bend.
    
    Args:
        step_53 (int): Absolute 53-TET step
    
    Returns:
        tuple: (midi_note, pitch_bend_value)
    """
    # Closest 12-TET MIDI note
    midi_note = round(step_53 * 12 / TET_53)
    midi_note = max(0, min(127, midi_note))
    
    # Residual in 53-TET steps
    expected_step = midi_note * TET_53 / 12.0
    residual_steps = step_53 - expected_step
    
    # Convert residual to cents, then to pitch bend
    residual_cents = residual_steps * 1200.0 / TET_53
    pitch_bend = int(round(residual_cents / PB_RANGE_CENTS * PB_MAX))
    pitch_bend = max(PB_MIN, min(PB_MAX, pitch_bend))
    
    return midi_note, pitch_bend


# =============================================================================
# MIDI MPE PARSER
# =============================================================================

def parse_mpe_midi(midi_path, speed=1.0):
    """
    Parse an MPE MIDI file into a list of chord events.
    
    Each chord event is a dict:
      {
        'onset_beats': float,   # onset time in beats
        'duration_beats': float, # duration in beats
        'notes': [
          {'step_53': int, 'velocity': int},
          ...
        ]
      }
    
    Args:
        midi_path: Path to MIDI file
        speed: Playback speed multiplier (default 1.0, no change)
    
    Returns:
        list[dict]: List of chord events, sorted by onset time
    """
    midi_path = Path(midi_path)
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    
    # We parse from the note track (usually track 1, or the main track)
    # Combine all tracks for safety
    channel_bends = {i: 0 for i in range(16)}  # raw pitch bend values
    active_notes = {}  # (channel, note) → {onset_ticks, step_53, velocity}
    
    # Collect all individual note events first
    note_events = []  # (onset_ticks, offset_ticks, step_53, velocity)
    
    # Parse each track
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            
            if msg.type == "pitchwheel":
                channel_bends[msg.channel] = msg.pitch
                
            elif msg.type == "note_on" and msg.velocity > 0:
                step_53 = midi_bend_to_53tet_step(msg.note, channel_bends.get(msg.channel, 0))
                active_notes[(msg.channel, msg.note)] = {
                    'onset': abs_time,
                    'step_53': step_53,
                    'velocity': msg.velocity
                }
                
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    info = active_notes.pop(key)
                    note_events.append({
                        'onset_ticks': info['onset'],
                        'offset_ticks': abs_time,
                        'step_53': info['step_53'],
                        'velocity': info['velocity']
                    })
    
    if not note_events:
        return []
    
    # Group simultaneous notes into chords
    # Notes with the same onset (within a small tolerance) form a chord
    note_events.sort(key=lambda x: (x['onset_ticks'], x['step_53']))
    
    chords = []
    current_onset = note_events[0]['onset_ticks']
    current_notes = []
    current_offset = 0
    
    TOLERANCE_TICKS = max(1, tpb // 48)  # ~20 ticks tolerance at 960 tpb
    
    for ne in note_events:
        if abs(ne['onset_ticks'] - current_onset) <= TOLERANCE_TICKS:
            current_notes.append(ne)
            current_offset = max(current_offset, ne['offset_ticks'])
        else:
            # Save previous chord
            if current_notes:
                onset_beats = current_onset / tpb / speed
                dur_beats = (current_offset - current_onset) / tpb / speed
                chords.append({
                    'onset_beats': round(onset_beats, 4),
                    'duration_beats': round(max(0.25, dur_beats), 4),
                    'notes': [
                        {'step_53': n['step_53'], 'velocity': n['velocity']}
                        for n in sorted(current_notes, key=lambda x: x['step_53'])
                    ]
                })
            # Start new chord
            current_onset = ne['onset_ticks']
            current_offset = ne['offset_ticks']
            current_notes = [ne]
    
    # Don't forget the last chord
    if current_notes:
        onset_beats = current_onset / tpb / speed
        dur_beats = (current_offset - current_onset) / tpb / speed
        chords.append({
            'onset_beats': round(onset_beats, 4),
            'duration_beats': round(max(0.25, dur_beats), 4),
            'notes': [
                {'step_53': n['step_53'], 'velocity': n['velocity']}
                for n in sorted(current_notes, key=lambda x: x['step_53'])
            ]
        })
    
    return chords


def clean_chords(chords, min_notes=3, max_tail_repeat=2):
    """
    Clean a chord progression for training.

    Two passes:
      1. **Merge undersized chords** — any chord with fewer than *min_notes*
         notes is absorbed into the nearest neighbour (by onset time).
      2. **Trim repetitive tails** — if the same chord (by pitch content)
         repeats more than *max_tail_repeat* times consecutively at the end,
         keep only *max_tail_repeat* occurrences.  Also trims ABAB
         alternating patterns the same way.

    Args:
        chords: list[dict] as returned by ``parse_mpe_midi``.
        min_notes: minimum number of notes for a chord to survive on its own.
        max_tail_repeat: maximum allowed consecutive identical (or alternating)
            chords at the end of the sequence.

    Returns:
        list[dict]: cleaned chord list (new list; originals are not mutated).
    """
    if not chords:
        return chords

    import copy
    out = copy.deepcopy(chords)

    # --- Pass 1: merge undersized chords ---
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            if len(out[i]['notes']) < min_notes and len(out) > 1:
                if i == 0:
                    target = i + 1
                elif i == len(out) - 1:
                    target = i - 1
                else:
                    d_prev = abs(out[i]['onset_beats'] - out[i - 1]['onset_beats'])
                    d_next = abs(out[i]['onset_beats'] - out[i + 1]['onset_beats'])
                    target = i - 1 if d_prev <= d_next else i + 1

                existing_steps = {n['step_53'] for n in out[target]['notes']}
                for n in out[i]['notes']:
                    if n['step_53'] not in existing_steps:
                        out[target]['notes'].append(n)
                        existing_steps.add(n['step_53'])
                out[target]['notes'].sort(key=lambda n: n['step_53'])

                out.pop(i)
                changed = True
            else:
                i += 1

    # --- Pass 2: trim repetitive tails ---
    if len(out) > max_tail_repeat:
        def _sig(chord):
            return tuple(sorted(n['step_53'] for n in chord['notes']))

        # 2a. Consecutive identical chords at the end
        last_sig = _sig(out[-1])
        tail_start = len(out) - 1
        while tail_start > 0 and _sig(out[tail_start - 1]) == last_sig:
            tail_start -= 1
        tail_len = len(out) - tail_start
        if tail_len > max_tail_repeat:
            out = out[:tail_start + max_tail_repeat]

        # 2b. ABAB alternating pattern at the end
        if len(out) >= max_tail_repeat * 2 + 2:
            sig_a = _sig(out[-2])
            sig_b = _sig(out[-1])
            if sig_a != sig_b:
                alt_count = 0
                for j in range(len(out) - 1, -1, -1):
                    expected = sig_b if (len(out) - 1 - j) % 2 == 0 else sig_a
                    if _sig(out[j]) == expected:
                        alt_count += 1
                    else:
                        break
                if alt_count > max_tail_repeat * 2:
                    keep = max_tail_repeat * 2
                    out = out[:len(out) - alt_count + keep]

    return out


# =============================================================================
# TOKENIZER
# =============================================================================

class MPETokenizer:
    """
    Tokenizer for 53-TET MPE MIDI files.
    
    Converts chord sequences to/from flat token sequences for GPT-2 training.
    
    Token format per chord:
      CHORD_START  DUR_<beats>  P_<step53> V_<bin>  ...  P_<step53> V_<bin>  CHORD_END
    
    Full sequence:
      <start>  [chord tokens...]  BAR  [chord tokens...]  ...  <end>
    
    Attributes:
        token_to_id (dict): Token string → integer ID
        id_to_token (dict): Integer ID → token string
        vocab_size (int): Total vocabulary size
    """
    
    def __init__(self, max_pitch=MAX_53TET_STEP, num_vel_bins=NUM_VELOCITY_BINS, 
                 duration_grid=None, beats_per_bar=4, pitch_offset=PITCH_OFFSET):
        """
        Initialize the tokenizer and build the vocabulary.
        
        Args:
            max_pitch: Maximum 53-TET step to include in vocabulary
            num_vel_bins: Number of velocity quantization bins
            duration_grid: List of allowed durations (beats). Uses default if None.
            beats_per_bar: Beats per bar for BAR token insertion (default 4)
            pitch_offset: Lowest 53-TET step to include in vocabulary
        """
        self.max_pitch = max_pitch
        self.num_vel_bins = num_vel_bins
        self.duration_grid = duration_grid or DURATION_GRID
        self.beats_per_bar = beats_per_bar
        self.pitch_offset = pitch_offset
        
        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Construct the full token vocabulary with deterministic ordering."""
        tokens = []
        
        # 1. Special tokens (IDs 0-3)
        tokens.extend([PAD_TOKEN, START_TOKEN, END_TOKEN, SEP_TOKEN])
        
        # 2. Structural tokens
        tokens.extend([CHORD_START_TOKEN, CHORD_END_TOKEN, BAR_TOKEN, REST_TOKEN])

        # 2a. Bar number tokens: BAR_1 … BAR_64
        #     Replace anonymous BAR — model learns position within song form
        tokens.extend(BAR_NUMBER_TOKENS)
        
        # 3. Duration tokens: DUR_<value>
        for dur in self.duration_grid:
            tokens.append(f"DUR_{dur}")
        
        # 4. Type conditioning tokens: TYPE_<label>
        #    14 transformation types from the 53-TET voicing system
        for label in TYPE_LABELS:
            tokens.append(f"{TYPE_PREFIX}_{label}")
        
        # 5. Style conditioning tokens: STYLE_<label>
        #    10 musical genre/style categories from iReal Pro metadata
        for label in STYLE_LABELS:
            tokens.append(f"{STYLE_PREFIX}_{label}")

        # 5b. Form conditioning tokens: FORM_<LABEL>
        #     Phrase-level structure markers (intro, A/B/C/D, coda, segno, ...)
        #     emitted at bar boundaries from iRealXML <rehearsal> data.
        tokens.extend(FORM_TOKENS)
        
        # 6. Root tokens: ROOT_<0..52> (53-TET pitch class of bass note)
        for r in range(TET_53):
            tokens.append(f"{ROOT_PREFIX}_{r}")
        
        # 7. Compound pitch+velocity tokens: PV_<step>_<vel_bin>
        #    Keeps intervallic contiguity within chords (no interleaved V_ tokens)
        for step in range(self.pitch_offset, self.max_pitch + 1):
            for v in range(1, self.num_vel_bins + 1):
                tokens.append(f"PV_{step}_{v}")
        
        # Build mappings
        self.token_to_id = {tok: i for i, tok in enumerate(tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(tokens)}
        self.vocab_size = len(tokens)
    
    # -----------------------------------------------------------------
    # Encoding: MIDI → Tokens
    # -----------------------------------------------------------------
    
    def encode_chords(self, chords, add_start_end=True, type_label=None, style_label=None,
                      form_markers=None):
        """
        Encode a list of chord events into a flat token sequence.
        
        The DUR token encodes the ONSET-TO-ONSET delta (harmonic rhythm),
        not the note-off duration. This is what matters for chord progression
        timing: how long until the NEXT chord arrives.
        
        For the last chord, we use its note-off duration since there's no
        next chord to compute a delta from.
        
        Args:
            chords: List of chord dicts from parse_mpe_midi()
            add_start_end: Whether to wrap with <start>/<end> tokens
            type_label: Transformation type label (e.g. "0_major", "2_subminor").
                        If provided, a TYPE_<label> token is prepended after <start>.
            style_label: Musical style label (e.g. "jazz", "blues").
                        If provided, a STYLE_<label> token is inserted after TYPE.
            form_markers: Optional dict {bar_idx_0based: form_label}. When a chord
                        opens a bar that has a form marker, a FORM_<LABEL> token is
                        emitted just before the corresponding BAR_<n> token (or at
                        the very start of the chord stream for bar 0).
        
        Returns:
            list[str]: Token string sequence
        """
        tokens = []

        if add_start_end:
            tokens.append(START_TOKEN)

        # Type conditioning token — tells the model which tuning system this is
        if type_label is not None:
            type_token = f"{TYPE_PREFIX}_{type_label}"
            if type_token in self.token_to_id:
                tokens.append(type_token)
            else:
                print(f"  Warning: unknown type label '{type_label}', skipping TYPE token")

        # Style conditioning token — tells the model which musical genre this is
        if style_label is not None:
            style_token = f"{STYLE_PREFIX}_{style_label}"
            if style_token in self.token_to_id:
                tokens.append(style_token)
            else:
                print(f"  Warning: unknown style label '{style_label}', skipping STYLE token")

        # Normalise form_markers into {bar_0idx:int -> FORM_<LABEL>:str}
        form_lookup = {}
        if form_markers:
            for bar_idx, raw in form_markers.items():
                canon = classify_form(raw)
                if canon is None:
                    continue
                ftok = f"{FORM_PREFIX}_{canon}"
                if ftok in self.token_to_id:
                    form_lookup[int(bar_idx)] = ftok

        # Emit FORM marker for bar 0 (first bar) before any chord, if present.
        if 0 in form_lookup:
            tokens.append(form_lookup[0])

        last_bar = -1  # Track bar lines (0-indexed)

        for i, chord in enumerate(chords):
            # Insert BAR_N token at bar boundaries (with optional FORM_X before it)
            current_bar = int(chord['onset_beats'] // self.beats_per_bar)  # 0-indexed
            if current_bar > last_bar:
                bars_to_emit = current_bar - max(0, last_bar)
                for b in range(bars_to_emit):
                    if last_bar >= 0:  # Don't add BAR before the first chord
                        bar_number = last_bar + b + 1 + 1  # 1-indexed
                        # FORM_ marker for the bar we are entering (0-indexed = bar_number - 1)
                        bar_0idx = bar_number - 1
                        if bar_0idx in form_lookup:
                            tokens.append(form_lookup[bar_0idx])
                        if bar_number <= BAR_MAX:
                            tokens.append(f"BAR_{bar_number}")
                        else:
                            tokens.append(BAR_TOKEN)
                last_bar = current_bar

            # Chord start
            tokens.append(CHORD_START_TOKEN)

            # Duration = onset-to-onset delta (harmonic rhythm), not note-off
            if i < len(chords) - 1:
                delta = chords[i + 1]['onset_beats'] - chord['onset_beats']
                q_dur = quantize_duration(max(0.5, delta))
            else:
                # Last chord: use its actual note-off duration
                q_dur = quantize_duration(chord['duration_beats'])
            tokens.append(f"DUR_{q_dur}")

            # Root = pitch class (mod 53) of the lowest note — explicit for the model
            sorted_notes = sorted(chord['notes'], key=lambda n: n['step_53'])[:MAX_CHORD_NOTES]
            if sorted_notes:
                root_pc = sorted_notes[0]['step_53'] % TET_53
                tokens.append(f"{ROOT_PREFIX}_{root_pc}")

            # Notes as compound PV tokens (sorted low→high, preserves interval contiguity)
            for note in sorted_notes:
                step = max(self.pitch_offset, min(self.max_pitch, note['step_53']))
                vel_bin = quantize_velocity(note['velocity'], self.num_vel_bins)
                tokens.append(f"PV_{step}_{vel_bin}")

            # Chord end
            tokens.append(CHORD_END_TOKEN)

        if add_start_end:
            tokens.append(END_TOKEN)

        return tokens
    
    def encode_file(self, midi_path, speed=1.0, add_start_end=True, type_label=None,
                    style_label=None, form_markers=None):
        """
        Parse and tokenize a MIDI MPE file.
        
        Args:
            midi_path: Path to MIDI file
            speed: Speed multiplier (applied to timing)
            add_start_end: Wrap with <start>/<end>
            type_label: Transformation type (e.g. "0_major") for conditioning.
                        If None, auto-detected from the file path (parent folder
                        name like "type_0_major" or filename suffix).
            style_label: Musical style (e.g. "jazz", "blues") for conditioning.
                        If None, no STYLE token is inserted.
        
        Returns:
            list[str]: Token sequence, or empty list on failure
        """
        chords = parse_mpe_midi(midi_path, speed=speed)
        if not chords:
            return []
        chords = clean_chords(chords)
        # Auto-detect type from path if not provided
        if type_label is None:
            type_label = _extract_type_label(midi_path)
        return self.encode_chords(chords, add_start_end=add_start_end,
                                  type_label=type_label, style_label=style_label,
                                  form_markers=form_markers)
    
    def encode_to_ids(self, tokens):
        """
        Convert token strings to integer IDs.
        
        Args:
            tokens: list[str] — token strings
        
        Returns:
            list[int]: Token IDs
        """
        pad_id = self.token_to_id[PAD_TOKEN]
        ids = []
        for t in tokens:
            tok_id = self.token_to_id.get(t)
            if tok_id is None:
                # Token not in vocab — likely a pitch outside the offset range.
                # Map to PAD but warn on first occurrence.
                if not hasattr(self, '_warned_tokens'):
                    self._warned_tokens = set()
                if t not in self._warned_tokens:
                    self._warned_tokens.add(t)
                    print(f"  Warning: token '{t}' not in vocabulary, mapping to <pad>")
                tok_id = pad_id
            ids.append(tok_id)
        return ids
    
    def decode_ids(self, ids):
        """
        Convert integer IDs back to token strings.
        
        Args:
            ids: list[int] — token IDs
        
        Returns:
            list[str]: Token strings
        """
        return [self.id_to_token.get(i, PAD_TOKEN) for i in ids]
    
    # -----------------------------------------------------------------
    # Decoding: Tokens → MIDI events
    # -----------------------------------------------------------------
    
    def decode(self, tokens):
        """
        Decode a token sequence back into chord events.
        
        Reconstructs chord events that can be written to a MIDI file.
        
        Args:
            tokens: list[str] — token sequence
        
        Returns:
            list[dict]: Chord events with 'onset_beats', 'duration_beats', 'notes'
        """
        chords = []
        current_beat = 0.0
        i = 0
        
        while i < len(tokens):
            tok = tokens[i]
            
            if tok == BAR_TOKEN or tok.startswith('BAR_'):
                # BAR / BAR_N is a structural marker for the model to learn phrase/bar
                # boundaries. It does NOT advance time — timing comes solely
                # from duration accumulation (onset-to-onset = duration in
                # our chord-per-beat dataset). This avoids the double-counting
                # bug where both BAR and duration would advance the clock.
                i += 1
                
            elif tok == CHORD_START_TOKEN:
                i += 1
                duration = 4.0  # default
                notes = []
                
                # Read chord contents until CHORD_END or end of sequence
                while i < len(tokens) and tokens[i] != CHORD_END_TOKEN:
                    t = tokens[i]
                    
                    if t.startswith("DUR_"):
                        duration = float(t[4:])
                    
                    elif t.startswith("PV_"):
                        # Compound pitch+velocity token: PV_<step>_<vel>
                        parts = t.split("_")
                        step_53 = int(parts[1])
                        vel_bin = int(parts[2])
                        vel = dequantize_velocity(vel_bin, self.num_vel_bins)
                        notes.append({'step_53': step_53, 'velocity': vel})
                    
                    elif t.startswith("P_"):
                        # Legacy P_/V_ pair support for decoding old data
                        step_53 = int(t[2:])
                        vel = 80  # default
                        # Look ahead for velocity
                        if i + 1 < len(tokens) and tokens[i + 1].startswith("V_"):
                            vel_bin = int(tokens[i + 1][2:])
                            vel = dequantize_velocity(vel_bin, self.num_vel_bins)
                            i += 1  # skip V_ token
                        notes.append({'step_53': step_53, 'velocity': vel})
                    
                    # ROOT_ tokens are metadata — not needed for MIDI reconstruction
                    
                    i += 1
                
                if notes:
                    chords.append({
                        'onset_beats': round(current_beat, 4),
                        'duration_beats': duration,
                        'notes': notes
                    })
                    current_beat += duration
                
                i += 1  # skip CHORD_END
            else:
                i += 1
        
        return chords
    
    # -----------------------------------------------------------------
    # MIDI Reconstruction
    # -----------------------------------------------------------------
    
    def chords_to_midi(self, chords, output_path, tpb=960, tempo_bpm=120):
        """
        Write chord events back to an MPE MIDI file.
        
        Args:
            chords: list[dict] — chord events (from decode())
            output_path: Path for the output .mid file
            tpb: Ticks per beat
            tempo_bpm: Tempo in BPM
        """
        mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
        
        # Track 0: tempo + RPN setup
        track0 = mido.MidiTrack()
        mid.tracks.append(track0)
        
        tempo = mido.bpm2tempo(tempo_bpm)
        track0.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        # Setup RPN for pitch bend range = 2 semitones on each channel
        for ch in range(1, 16):
            track0.append(mido.Message('control_change', channel=ch, control=101, value=0, time=0))
            track0.append(mido.Message('control_change', channel=ch, control=100, value=0, time=0))
            track0.append(mido.Message('control_change', channel=ch, control=6, value=2, time=0))
            track0.append(mido.Message('control_change', channel=ch, control=38, value=0, time=0))
            track0.append(mido.Message('control_change', channel=ch, control=101, value=127, time=0))
            track0.append(mido.Message('control_change', channel=ch, control=100, value=127, time=0))
        
        # Track 1: notes
        track1 = mido.MidiTrack()
        mid.tracks.append(track1)
        
        # Also set up RPN on track 1
        for ch in range(1, 16):
            track1.append(mido.Message('control_change', channel=ch, control=101, value=0, time=0))
            track1.append(mido.Message('control_change', channel=ch, control=100, value=0, time=0))
            track1.append(mido.Message('control_change', channel=ch, control=6, value=2, time=0))
            track1.append(mido.Message('control_change', channel=ch, control=38, value=0, time=0))
            track1.append(mido.Message('control_change', channel=ch, control=101, value=127, time=0))
            track1.append(mido.Message('control_change', channel=ch, control=100, value=127, time=0))
        
        # Collect all note_on/note_off events with absolute times
        events = []
        channel_pool = list(range(1, 16))  # channels 1-15 for MPE
        
        for chord in chords:
            onset_ticks = int(chord['onset_beats'] * tpb)
            offset_ticks = int((chord['onset_beats'] + chord['duration_beats']) * tpb)
            
            for j, note in enumerate(chord['notes']):
                ch = channel_pool[j % len(channel_pool)]
                midi_note, pitch_bend = step53_to_midi_and_bend(note['step_53'])
                vel = note.get('velocity', 80)
                
                # Pitch bend before note_on
                events.append((onset_ticks, 'pitchwheel', ch, pitch_bend, 0))
                events.append((onset_ticks, 'note_on', ch, midi_note, vel))
                events.append((offset_ticks, 'note_off', ch, midi_note, vel))
        
        # Sort by time, with note_off before note_on at same time, pitchwheel before note_on
        priority = {'pitchwheel': 0, 'note_off': 1, 'note_on': 2}
        events.sort(key=lambda e: (e[0], priority.get(e[1], 1)))
        
        # Convert to delta-time messages
        last_time = 0
        for ev in events:
            abs_time, msg_type, ch, val1, val2 = ev
            delta = abs_time - last_time
            
            if msg_type == 'pitchwheel':
                track1.append(mido.Message('pitchwheel', channel=ch, pitch=val1, time=delta))
            elif msg_type == 'note_on':
                track1.append(mido.Message('note_on', channel=ch, note=val1, velocity=val2, time=delta))
            elif msg_type == 'note_off':
                track1.append(mido.Message('note_off', channel=ch, note=val1, velocity=val2, time=delta))
            
            last_time = abs_time
        
        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(output_path)
    
    # -----------------------------------------------------------------
    # Padding & Batching
    # -----------------------------------------------------------------
    
    def pad_sequence(self, token_ids, max_length):
        """
        Pad or truncate a token ID sequence to a fixed length.
        
        Args:
            token_ids: list[int] — token IDs
            max_length: Target sequence length
        
        Returns:
            list[int]: Padded/truncated sequence
        """
        pad_id = self.token_to_id[PAD_TOKEN]
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        return token_ids + [pad_id] * (max_length - len(token_ids))
    
    # -----------------------------------------------------------------
    # Vocabulary I/O
    # -----------------------------------------------------------------
    
    def save_vocab(self, path):
        """Save vocabulary to JSON file."""
        data = {
            'token_to_id': self.token_to_id,
            'config': {
                'max_pitch': self.max_pitch,
                'num_vel_bins': self.num_vel_bins,
                'duration_grid': self.duration_grid,
                'beats_per_bar': self.beats_per_bar,
                'pitch_offset': self.pitch_offset,
                'vocab_size': self.vocab_size
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_vocab(cls, path):
        """Load tokenizer from a saved vocabulary JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = data['config']
        tokenizer = cls(
            max_pitch=config['max_pitch'],
            num_vel_bins=config['num_vel_bins'],
            duration_grid=config['duration_grid'],
            beats_per_bar=config['beats_per_bar'],
            pitch_offset=config.get('pitch_offset', 0)
        )
        return tokenizer
    
    # -----------------------------------------------------------------
    # Statistics & Debugging
    # -----------------------------------------------------------------
    
    def describe(self):
        """Print a summary of the tokenizer vocabulary."""
        print("=" * 60)
        print("MPE 53-TET Tokenizer — Vocabulary Summary")
        print("=" * 60)
        
        # Count token types
        n_special = 4
        n_structural = 4
        n_duration = len(self.duration_grid)
        n_type = len(TYPE_LABELS)
        n_root = TET_53
        n_pitch = self.max_pitch - self.pitch_offset + 1
        n_pv = n_pitch * self.num_vel_bins
        
        print(f"  Total vocab size:   {self.vocab_size}")
        print(f"  Special tokens:     {n_special}  (IDs 0-{n_special-1})")
        print(f"  Structural tokens:  {n_structural}  ({CHORD_START_TOKEN}, {CHORD_END_TOKEN}, {BAR_TOKEN}, {REST_TOKEN})")
        print(f"  Duration tokens:    {n_duration}  (DUR_{self.duration_grid[0]} .. DUR_{self.duration_grid[-1]})")
        print(f"  Type tokens:        {n_type}  ({TYPE_TOKENS[0]} .. {TYPE_TOKENS[-1]})")
        print(f"  Root tokens:        {n_root}  (ROOT_0 .. ROOT_52)")
        print(f"  PV tokens:          {n_pv}  (PV_{self.pitch_offset}_1 .. PV_{self.max_pitch}_{self.num_vel_bins})")
        print(f"  Beats per bar:      {self.beats_per_bar}")
        print("=" * 60)
    
    def token_stats(self, tokens):
        """
        Print distribution statistics for a token sequence.
        
        Args:
            tokens: list[str] — token sequence
        """
        counts = Counter(tokens)
        
        # Group by category
        pitches = {k: v for k, v in counts.items() if k.startswith("P_")}
        durations = {k: v for k, v in counts.items() if k.startswith("DUR_")}
        velocities = {k: v for k, v in counts.items() if k.startswith("V_")}
        
        print(f"Total tokens: {len(tokens)}")
        print(f"Unique tokens used: {len(counts)}")
        print(f"Chords: {counts.get(CHORD_START_TOKEN, 0)}")
        print(f"Bars: {counts.get(BAR_TOKEN, 0)}")
        print(f"\nDuration distribution:")
        for k, v in sorted(durations.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print(f"\nVelocity distribution:")
        for k, v in sorted(velocities.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print(f"\nPitch range: {min(pitches.keys())} — {max(pitches.keys())} ({len(pitches)} unique)")


# =============================================================================
# DATASET CLASS (for GPT-2 training)
# =============================================================================

class MPETokenDataset:
    """
    PyTorch-compatible dataset that tokenizes MIDI MPE files for GPT-2 training.
    
    Each item is a pair (x, y) where:
      x = token_ids[:-1]  (input)
      y = token_ids[1:]   (target, shifted by 1)
    
    This follows the standard autoregressive language model training pattern.
    
    Usage:
        tokenizer = MPETokenizer()
        dataset = MPETokenDataset(
            midi_dir="dataset/midi_files/53_tet_mpe",
            tokenizer=tokenizer,
            block_size=512
        )
        # Use with PyTorch DataLoader
    """
    
    def __init__(self, midi_dir, tokenizer, block_size=512, max_files=None, 
                 file_pattern="*.mid", speed=1.0, verbose=True):
        """
        Args:
            midi_dir: Directory containing MIDI files
            tokenizer: MPETokenizer instance
            block_size: Sequence length for training (context window)
            max_files: Max files to load (None = all)
            file_pattern: Glob pattern for MIDI files
            speed: Speed multiplier for timing
            verbose: Print progress
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.vocab_size = tokenizer.vocab_size
        
        midi_dir = Path(midi_dir)
        files = sorted(midi_dir.glob(file_pattern))
        
        if max_files:
            files = files[:max_files]
        
        if verbose:
            print(f"Loading {len(files)} MIDI files from {midi_dir}...")
        
        # Tokenize all files and collect sequences
        self.sequences = []
        failed = 0
        
        for f in files:
            try:
                tokens = tokenizer.encode_file(f, speed=speed)
                if tokens:
                    ids = tokenizer.encode_to_ids(tokens)
                    self.sequences.append(ids)
            except Exception as e:
                failed += 1
                if verbose and failed <= 5:
                    print(f"  Warning: failed to parse {f.name}: {e}")
        
        if verbose:
            print(f"Successfully tokenized: {len(self.sequences)} files ({failed} failed)")
            lengths = [len(s) for s in self.sequences]
            if lengths:
                print(f"Sequence lengths — min: {min(lengths)}, max: {max(lengths)}, "
                      f"mean: {sum(lengths)/len(lengths):.0f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns (x, y) tensors for autoregressive training.
        
        x = padded_ids[:-1]  (input)
        y = padded_ids[1:]   (shifted target)
        """
        ids = self.tokenizer.pad_sequence(self.sequences[idx], self.block_size + 1)
        ids = _torch().tensor(ids, dtype=_torch().long)
        
        x = ids[:-1]  # input:  [0, 1, 2, ..., block_size-1]
        y = ids[1:]   # target: [1, 2, 3, ..., block_size]
        
        return x, y


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def tokenize_dataset(midi_dir, output_dir, tokenizer=None, max_files=None, speed=1.0):
    """
    Batch-tokenize all MIDI files in a directory and save as JSON.
    
    Saves:
      - <output_dir>/tokenized_sequences.json  (all token ID sequences)
      - <output_dir>/vocab.json                 (tokenizer vocabulary)
      - <output_dir>/stats.json                 (dataset statistics)
    
    Args:
        midi_dir: Input directory with .mid files
        output_dir: Output directory
        tokenizer: MPETokenizer (creates default if None)
        max_files: Max files to process
        speed: Speed multiplier
    
    Returns:
        tuple: (tokenizer, sequences, stats)
    """
    if tokenizer is None:
        tokenizer = MPETokenizer()
    
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(midi_dir.glob("*.mid"))
    if max_files:
        files = files[:max_files]
    
    print(f"Tokenizing {len(files)} files...")
    
    sequences = []
    all_tokens = []
    failed = 0
    
    for i, f in enumerate(files):
        try:
            tokens = tokenizer.encode_file(f, speed=speed)
            if tokens:
                ids = tokenizer.encode_to_ids(tokens)
                sequences.append({
                    'file': f.name,
                    'token_ids': ids,
                    'length': len(ids)
                })
                all_tokens.extend(tokens)
        except Exception as e:
            failed += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(files)} ({failed} failed)")
    
    print(f"Done. {len(sequences)} sequences, {failed} failures")
    
    # Statistics
    lengths = [s['length'] for s in sequences]
    token_counts = Counter(all_tokens)
    
    stats = {
        'total_files': len(files),
        'successful': len(sequences),
        'failed': failed,
        'total_tokens': len(all_tokens),
        'unique_tokens_used': len(token_counts),
        'vocab_size': tokenizer.vocab_size,
        'seq_length_min': min(lengths) if lengths else 0,
        'seq_length_max': max(lengths) if lengths else 0,
        'seq_length_mean': round(sum(lengths) / len(lengths), 1) if lengths else 0,
        'top_20_tokens': token_counts.most_common(20)
    }
    
    # Save outputs
    tokenizer.save_vocab(output_dir / "vocab.json")
    
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save sequences (token IDs only, for efficiency)
    seq_data = [{'file': s['file'], 'ids': s['token_ids']} for s in sequences]
    with open(output_dir / "tokenized_sequences.json", 'w') as f:
        json.dump(seq_data, f)
    
    print(f"\nSaved to {output_dir}/")
    print(f"  vocab.json ({tokenizer.vocab_size} tokens)")
    print(f"  tokenized_sequences.json ({len(sequences)} sequences)")
    print(f"  stats.json")
    
    return tokenizer, sequences, stats


# =============================================================================
# ROUNDTRIP VERIFICATION
# =============================================================================

def verify_roundtrip(midi_path, output_path=None, tokenizer=None, verbose=True):
    """
    Verify encode → decode → MIDI roundtrip for a single file.
    
    Encodes a MIDI file to tokens, decodes back to chords, and optionally
    writes a new MIDI file. Reports any discrepancies.
    
    Args:
        midi_path: Path to source MIDI file
        output_path: Path for reconstructed MIDI (optional)
        tokenizer: MPETokenizer (creates default if None)
        verbose: Print details
    
    Returns:
        dict: Comparison report
    """
    if tokenizer is None:
        tokenizer = MPETokenizer()
    
    # Step 1: Parse original
    original_chords = parse_mpe_midi(midi_path)
    
    # Step 2: Encode
    tokens = tokenizer.encode_chords(original_chords)
    token_ids = tokenizer.encode_to_ids(tokens)
    
    # Step 3: Decode back
    decoded_tokens = tokenizer.decode_ids(token_ids)
    reconstructed_chords = tokenizer.decode(decoded_tokens)
    
    # Step 4: Compare
    report = {
        'file': str(midi_path),
        'original_chords': len(original_chords),
        'reconstructed_chords': len(reconstructed_chords),
        'token_count': len(tokens),
        'match': len(original_chords) == len(reconstructed_chords)
    }
    
    if verbose:
        print(f"Roundtrip: {Path(midi_path).name}")
        print(f"  Original chords:      {len(original_chords)}")
        print(f"  Token sequence length: {len(tokens)}")
        print(f"  Reconstructed chords:  {len(reconstructed_chords)}")
        
        # Compare pitch content
        if original_chords and reconstructed_chords:
            n_compare = min(len(original_chords), len(reconstructed_chords))
            pitch_matches = 0
            for i in range(n_compare):
                orig_pitches = sorted([n['step_53'] for n in original_chords[i]['notes']])
                recon_pitches = sorted([n['step_53'] for n in reconstructed_chords[i]['notes']])
                if orig_pitches == recon_pitches:
                    pitch_matches += 1
            
            accuracy = pitch_matches / n_compare * 100
            report['pitch_accuracy'] = round(accuracy, 1)
            print(f"  Pitch accuracy:       {accuracy:.1f}% ({pitch_matches}/{n_compare})")
    
    # Step 5: Write reconstructed MIDI
    if output_path and reconstructed_chords:
        tokenizer.chords_to_midi(reconstructed_chords, output_path)
        if verbose:
            print(f"  Saved: {output_path}")
    
    return report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="53-TET MPE MIDI Tokenizer")
    subparsers = parser.add_subparsers(dest="command")
    
    # Describe vocabulary
    sub = subparsers.add_parser("describe", help="Show tokenizer vocabulary info")
    
    # Encode a single file
    sub = subparsers.add_parser("encode", help="Tokenize a single MIDI file")
    sub.add_argument("midi_file", help="Path to MIDI file")
    sub.add_argument("--stats", action="store_true", help="Show token statistics")
    
    # Verify roundtrip
    sub = subparsers.add_parser("verify", help="Verify encode/decode roundtrip")
    sub.add_argument("midi_file", help="Path to MIDI file")
    sub.add_argument("--output", "-o", help="Output MIDI path for comparison")
    
    # Batch tokenize
    sub = subparsers.add_parser("batch", help="Batch tokenize a directory")
    sub.add_argument("midi_dir", help="Directory with MIDI files")
    sub.add_argument("output_dir", help="Output directory for tokenized data")
    sub.add_argument("--max-files", type=int, help="Max files to process")
    
    args = parser.parse_args()
    tokenizer = MPETokenizer()
    
    if args.command == "describe":
        tokenizer.describe()
    
    elif args.command == "encode":
        tokens = tokenizer.encode_file(args.midi_file)
        if tokens:
            print(f"Tokens ({len(tokens)}):")
            print(" ".join(tokens[:80]))
            if len(tokens) > 80:
                print(f"  ... ({len(tokens) - 80} more)")
            if args.stats:
                print()
                tokenizer.token_stats(tokens)
        else:
            print("Failed to tokenize file.")
    
    elif args.command == "verify":
        verify_roundtrip(args.midi_file, args.output, tokenizer)
    
    elif args.command == "batch":
        tokenize_dataset(args.midi_dir, args.output_dir, tokenizer, args.max_files)
    
    else:
        parser.print_help()
