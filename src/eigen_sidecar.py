"""
eigen_sidecar.py
================
Compute per-chord EigenSpace sidecars for the paired 53-TET dataset.

For each paired (``.mid``, ``.txt``) song, this module produces a
``<stem>.eigen.npy`` file of shape ``(N_chords, 4)`` and dtype ``float32``.
Column order: ``(α, β, γ, D)``.

Semantics (locked by STRATEGY_V3 §5):

* **α, β, γ** — frequency ratios of the chord's 3rd / 5th / 7th relative to
  the root, computed on octave-folded interval classes. Root-invariant:
  ``Cmaj7 ≡ Dmaj7 ≡ F#maj7``.
* **D** — scalar dissonance value from the pre-computed 4D dissonance map
  (``eigenspace.DissonanceMap.lookup(α, β, γ)``). Also root-invariant.
* Root reference is the **symbolic chord root** from the L1 sidecar (e.g.
  ``R_C`` — the name the chord is **called**, not its bass). Slash chords
  use the base configuration (``Cmaj7/E`` ≡ ``Cmaj7``).
* Pitches are folded into one octave before interval classification.
  9ths / 11ths / 13ths collapse to their parent classes; notes that fall
  outside the α/β/γ zones after folding are discarded as extensions.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable

from chord_mapping import NOTE_NAMES_53TET
from eigenspace import DissonanceMap, classify_intervals
import l1
import tokenizer as _tokenizer_module


# Index lookup: name → 53-TET pitch class (0..52)
_NAME_TO_PC: dict[str, int] = {name: i for i, name in enumerate(NOTE_NAMES_53TET)}


def root_name_to_pc(name: str) -> int:
    """Map an L1 root name (e.g. ``C``, ``vG#``, ``^^Eb``) to 53-TET pc (0..52).

    Raises KeyError if the name is not in NOTE_NAMES_53TET.
    """
    return _NAME_TO_PC[name]


def chord_eigen_from_pitches(
    pitches_53: Iterable[int],
    root_pc: int,
    diss_map: DissonanceMap,
) -> tuple[float, float, float, float]:
    """Compute ``(α, β, γ, D)`` for a single chord.

    Parameters
    ----------
    pitches_53 : iterable of int
        Absolute 53-TET step for each note in the chord (MIDI-derived).
    root_pc : int
        53-TET pitch class of the chord's **symbolic** root (0..52). For
        slash chords, use the named root, not the bass.
    diss_map : DissonanceMap
        Pre-computed 4D dissonance map; its ``.lookup`` yields ``D``.

    Returns
    -------
    (alpha, beta, gamma, D) : tuple[float, float, float, float]
    """
    # Fold every pitch into the chord's root octave as an interval class.
    intervals = set()
    for p in pitches_53:
        ic = (int(p) - root_pc) % 53
        if ic != 0:
            intervals.add(ic)

    # Zone classifier: picks 3rd / 5th / 7th and returns ratios (root-invariant).
    alpha, beta, gamma = classify_intervals(sorted(intervals))

    d = diss_map.lookup(alpha, beta, gamma)
    if d is None:
        d = float(diss_map.diss_mean)  # safe fallback; α/β/γ stay valid
    return alpha, beta, gamma, float(d)


# ---------------------------------------------------------------------------
# Per-song sidecar builder
# ---------------------------------------------------------------------------


def build_eigen_sidecar(
    midi_path: str | Path,
    text_path: str | Path,
    diss_map: DissonanceMap,
) -> np.ndarray:
    """Compute the ``(N_chords, 4)`` EigenSpace array for one paired song.

    The chord count must match between L1 (text sidecar) and L2 (MIDI);
    mismatch raises via :func:`l1.merge_levels` upstream logic.
    """
    # L1 — symbolic chord names (authoritative source for the root)
    parsed = l1.parse_text_sidecar(
        text_path,
        classify_style_fn=_tokenizer_module.classify_style,
    )
    chord_events = [e for e in parsed.events if isinstance(e, l1.ChordEvent)]

    # L2 — MIDI pitches per chord block (no clean_chords, preserves 1:1)
    chords = _tokenizer_module.parse_mpe_midi(str(midi_path))
    if len(chords) != len(chord_events):
        raise ValueError(
            f"L1/L2 chord-count mismatch in {Path(midi_path).name}: "
            f"L1={len(chord_events)} L2={len(chords)}"
        )

    out = np.zeros((len(chord_events), 4), dtype=np.float32)
    for k, (ev, mc) in enumerate(zip(chord_events, chords)):
        try:
            root_pc = root_name_to_pc(ev.root)
        except KeyError:
            # Unknown root (should not happen given the frozen alphabet);
            # fall back to the lowest MIDI pitch's pc.
            root_pc = int(mc["notes"][0]["step_53"]) % 53
        pitches = [n["step_53"] for n in mc["notes"]]
        alpha, beta, gamma, d = chord_eigen_from_pitches(pitches, root_pc, diss_map)
        out[k] = (alpha, beta, gamma, d)
    return out


def save_sidecar(arr: np.ndarray, midi_path: str | Path) -> Path:
    """Write ``arr`` to ``<midi_stem>.eigen.npy`` next to the MIDI file."""
    p = Path(midi_path)
    out_path = p.with_suffix(".eigen.npy")
    np.save(out_path, arr)
    return out_path
