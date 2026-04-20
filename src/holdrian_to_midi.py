"""
holdrian_to_midi.py
===================
Deterministic Method-B translator — pure 53-EDO holdrian token stream →
MPE MIDI file. The inverse of ``tokenizer_b.TokenizerB.encode_chords``
combined with ``MPETokenizer.chords_to_midi``.

No learning happens here. This is the post-hoc lookup described in
``STRATEGY_V3.md`` §B.2:

    H_<step>_<vel>  →  (nearest MIDI note, pitch-bend) via
                        ``tokenizer.step53_to_midi_and_bend``

The MPE setup (RPN ±2 semitones on every channel, one note per channel
with its own pitch-bend) is reused verbatim from Method A — pitch-bend
math is a MIDI-level detail, not a Method-A semantic claim, so it is
shared identically across both methods. This keeps the A↔B comparison
honest at render time.

Public API
----------
  tokens_to_chords(tokens: list[str], tokenizer=None) -> list[dict]
      Decode a Method-B token stream to chord events
      ``{'onset_beats', 'duration_beats', 'notes': [{step_53, velocity}]}``.

  holdrian_to_midi(tokens: list[str] | list[int], output_path, *,
                   tokenizer=None, tpb=960, tempo_bpm=120) -> list[dict]
      Full pipeline: token-ids or token-strings → MIDI file on disk.
      Returns the decoded chord list for inspection.

Isolation contract (STRATEGY_V3.md §B.0):
  * New file, ``_b`` is implicit in the name (holdrian is Method-B only).
  * Imports ``tokenizer`` and ``tokenizer_b`` read-only.
  * No edits to ``tokenizer.py`` or ``chords_to_midi``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Read-only imports of frozen Method-A helpers and the Method-B tokenizer.
from tokenizer import (
    MPETokenizer,
    step53_to_midi_and_bend,  # kept as the canonical 53-EDO → (midi, bend) map
)
from tokenizer_b import TokenizerB


# ---------------------------------------------------------------------------
# Token → chord events
# ---------------------------------------------------------------------------


def tokens_to_chords(
    tokens: Iterable,
    tokenizer: TokenizerB | None = None,
) -> list[dict]:
    """Decode a Method-B token stream into chord events.

    Parameters
    ----------
    tokens
        Either a list of token *strings* (e.g. ``["<start>", "CHORD_START",
        "DUR_4.0", "H_212_5", ...]``) or a list of integer ids. Ids are
        converted back to strings via ``tokenizer.decode_ids``.
    tokenizer
        Optional :class:`TokenizerB`. A fresh one is constructed if not
        supplied. Only needed when ``tokens`` contains integer ids.

    Returns
    -------
    list[dict]
        Chord events with the same schema as ``MPETokenizer.decode``:
        ``{'onset_beats': float, 'duration_beats': float,
        'notes': [{'step_53': int, 'velocity': int}, ...]}``.
        ``step_53`` is a **holdrian comma index**, not a MIDI note.
    """
    tokens = list(tokens)
    if not tokens:
        return []

    if isinstance(tokens[0], int) or (isinstance(tokens[0], str) and tokens[0].isdigit()):
        tok = tokenizer or TokenizerB()
        tokens = tok.decode_ids([int(t) for t in tokens])
    else:
        tok = tokenizer or TokenizerB()

    return tok.decode(tokens)


# ---------------------------------------------------------------------------
# Chord events → MPE MIDI
# ---------------------------------------------------------------------------


def _render_mpe_midi(chords: list[dict], output_path, tpb: int, tempo_bpm: int) -> None:
    """Write chord events to an MPE MIDI file.

    This is a direct port of :meth:`MPETokenizer.chords_to_midi` kept local
    to avoid depending on a concrete Method-A tokenizer *instance*: Method
    B's generated stream is rendered with the same MPE RPN setup, pitch
    bend per channel, and delta-time encoding as Method A — the choice is
    deliberate so A↔B audio outputs differ only through their vocabulary.
    """
    import mido

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    track0 = mido.MidiTrack(); mid.tracks.append(track0)
    track0.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm), time=0))
    for ch in range(1, 16):
        track0.append(mido.Message('control_change', channel=ch, control=101, value=0, time=0))
        track0.append(mido.Message('control_change', channel=ch, control=100, value=0, time=0))
        track0.append(mido.Message('control_change', channel=ch, control=6,   value=2, time=0))
        track0.append(mido.Message('control_change', channel=ch, control=38,  value=0, time=0))
        track0.append(mido.Message('control_change', channel=ch, control=101, value=127, time=0))
        track0.append(mido.Message('control_change', channel=ch, control=100, value=127, time=0))

    track1 = mido.MidiTrack(); mid.tracks.append(track1)
    for ch in range(1, 16):
        track1.append(mido.Message('control_change', channel=ch, control=101, value=0, time=0))
        track1.append(mido.Message('control_change', channel=ch, control=100, value=0, time=0))
        track1.append(mido.Message('control_change', channel=ch, control=6,   value=2, time=0))
        track1.append(mido.Message('control_change', channel=ch, control=38,  value=0, time=0))
        track1.append(mido.Message('control_change', channel=ch, control=101, value=127, time=0))
        track1.append(mido.Message('control_change', channel=ch, control=100, value=127, time=0))

    events = []
    channel_pool = list(range(1, 16))
    for chord in chords:
        onset_ticks = int(chord['onset_beats'] * tpb)
        offset_ticks = int((chord['onset_beats'] + chord['duration_beats']) * tpb)
        for j, note in enumerate(chord['notes']):
            ch = channel_pool[j % len(channel_pool)]
            midi_note, pitch_bend = step53_to_midi_and_bend(note['step_53'])
            vel = int(note.get('velocity', 80))
            events.append((onset_ticks, 'pitchwheel', ch, pitch_bend, 0))
            events.append((onset_ticks, 'note_on', ch, midi_note, vel))
            events.append((offset_ticks, 'note_off', ch, midi_note, vel))

    priority = {'pitchwheel': 0, 'note_off': 1, 'note_on': 2}
    events.sort(key=lambda e: (e[0], priority.get(e[1], 1)))

    last_time = 0
    for abs_time, msg_type, ch, val1, val2 in events:
        delta = abs_time - last_time
        if msg_type == 'pitchwheel':
            track1.append(mido.Message('pitchwheel', channel=ch, pitch=val1, time=delta))
        elif msg_type == 'note_on':
            track1.append(mido.Message('note_on', channel=ch, note=val1, velocity=val2, time=delta))
        elif msg_type == 'note_off':
            track1.append(mido.Message('note_off', channel=ch, note=val1, velocity=val2, time=delta))
        last_time = abs_time

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)


# ---------------------------------------------------------------------------
# Top-level translator
# ---------------------------------------------------------------------------


def holdrian_to_midi(
    tokens: Iterable,
    output_path,
    *,
    tokenizer: TokenizerB | None = None,
    tpb: int = 960,
    tempo_bpm: int = 120,
) -> list[dict]:
    """Translate a Method-B token stream to an MPE MIDI file on disk.

    Parameters
    ----------
    tokens
        List of token strings or integer ids produced by ``generate_b``.
    output_path
        Destination ``.mid`` file. Parent dirs are created.
    tokenizer
        Optional :class:`TokenizerB` (only required when ``tokens`` contains ids).
    tpb, tempo_bpm
        MIDI ticks-per-beat and tempo.

    Returns
    -------
    list[dict]
        The decoded chord events, useful for inspection / logging.
    """
    chords = tokens_to_chords(tokens, tokenizer=tokenizer)
    _render_mpe_midi(chords, output_path, tpb=tpb, tempo_bpm=tempo_bpm)
    return chords


__all__ = ["tokens_to_chords", "holdrian_to_midi"]
