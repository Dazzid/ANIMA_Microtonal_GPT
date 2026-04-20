"""
l1.py
=====
Level-1 symbolic alphabet and text-sidecar parser.

L1 tokens come from the `.txt` sidecars in
``dataset/text_files/53_tet_mpe/<type>/<stem>.txt``. They describe each
chord symbolically (style, tonality, form, structural markers, duration,
root, quality, extensions, slash bass). L2 tokens (MIDI+MPE) are handled
by :mod:`tokenizer`.

The alphabet is frozen in ``dataset/l1_alphabet.json`` (produced by a full
scan of the 672,840 sidecars). This module loads that file, exposes the
deterministic vocabulary lists, and parses sidecars into structured events
that :func:`merge_levels` can align with L2 chord blocks.

Token prefixes
--------------
- ``R_<name>``   — root or slash bass (e.g. ``R_C``, ``R_vG#``, ``R_^^Eb``)
- ``Q_<qual>``   — chord quality (e.g. ``Q_maj7``, ``Q_vMvM7``,
  ``Q_maj_implicit`` for an empty-string quality in the sidecar)
- ``X_<phrase>`` — joined extension phrase (e.g. ``X_add 9``, ``X_alter b5``)
- ``TONALITY_<key>_<mode>`` — e.g. ``TONALITY_C_major``
- Structural: ``.``, ``|``, ``|:``, ``:|``, ``/``, ``<style>``, ``<tonality>``
- Form: ``FORM_A``, ``FORM_B``, ... (already defined in :mod:`tokenizer`)
- Type: ``TYPE_<label>`` (already defined)
- Style: ``STYLE_<label>`` (already defined; mapped via
  ``formats.correctStyleTokensInMeta``)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Duration quantization grid (kept in sync with tokenizer.DURATION_GRID).
# Sidecars store raw floats (e.g. "1.333...") that must snap to this grid
# so the emitted DUR_* tokens exist in the vocab.
_DURATION_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def _quantize_duration_str(raw) -> str:
    """Snap a raw duration (number or string) to DURATION_GRID.

    Returns the grid value as a plain float-string (matches tokenizer's
    ``f"DUR_{dur}"`` format so lookups succeed).
    """
    try:
        d = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    if d <= 0:
        d = _DURATION_GRID[0]
    best = min(_DURATION_GRID, key=lambda g: abs(d - g))
    return str(best)


# ---------------------------------------------------------------------------
# Alphabet loading
# ---------------------------------------------------------------------------

ALPHABET_PATH = (
    Path(__file__).resolve().parent.parent / "dataset" / "l1_alphabet.json"
)

# Sidecar strings that are NOT chord content.
_SIDECAR_STRUCTURAL = {
    "|", "|:", ":|", "||", "b||", "e||",
    ".", "/",
    "<style>", "<tonality>",
}
_SIDECAR_FORM_RAW = {
    "Form_A", "Form_B", "Form_C", "Form_D",
    "Form_Head", "Form_Coda", "Form_Segno",
    "Form_intro", "Form_verse",
    "Intro",
}
_SIDECAR_REPEAT_RAW = {f"Repeat_{i}" for i in range(8)}

# Map raw form markers in sidecars to canonical FORM_<LABEL> tokens used in
# the tokenizer vocabulary.
FORM_RAW_TO_CANON = {
    "Form_A": "FORM_A",
    "Form_B": "FORM_B",
    "Form_C": "FORM_C",
    "Form_D": "FORM_D",
    "Form_Head": "FORM_HEAD",
    "Form_Coda": "FORM_CODA",
    "Form_Segno": "FORM_SEGNO",
    "Form_intro": "FORM_INTRO",
    "Form_verse": "FORM_VERSE",
    "Intro": "FORM_INTRO",
}

# Empty quality in sidecar ('' = implicit major triad) → explicit token.
EMPTY_QUALITY_TOKEN = "Q_maj_implicit"


@dataclass(frozen=True)
class L1Alphabet:
    roots: tuple[str, ...]
    qualities: tuple[str, ...]   # excludes '' — stored separately as EMPTY_QUALITY_TOKEN
    extensions: tuple[str, ...]
    slash_bases: tuple[str, ...]  # subset of roots in practice; kept for clarity
    tonalities: tuple[str, ...]   # key_mode strings, e.g. "C_major"

    def vocab_tokens(self) -> list[str]:
        """Return deterministic list of L1 tokens to add to the tokenizer vocab."""
        toks: list[str] = []
        # Structural (low-frequency, fixed)
        toks.extend(["<style>", "<tonality>", ".", "|", "|:", ":|", "/"])
        # Tonalities
        toks.extend(f"TONALITY_{t}" for t in self.tonalities)
        # Roots (also covers slash bases since those are a subset)
        all_roots = tuple(sorted(set(self.roots) | set(self.slash_bases)))
        toks.extend(f"R_{r}" for r in all_roots)
        # Qualities
        toks.append(EMPTY_QUALITY_TOKEN)
        toks.extend(f"Q_{q}" for q in self.qualities)
        # Extensions (joined phrases, space preserved)
        toks.extend(f"X_{e}" for e in self.extensions)
        return toks


def load_alphabet(path: str | Path | None = None) -> L1Alphabet:
    """Load the frozen L1 alphabet from JSON and return a deterministic object."""
    p = Path(path) if path is not None else ALPHABET_PATH
    data = json.loads(p.read_text())

    roots = tuple(sorted(data.get("roots", {}).keys()))
    # Drop empty string from qualities — handled by EMPTY_QUALITY_TOKEN
    qualities_raw = [q for q in data.get("qualities", {}) if q != ""]
    qualities = tuple(sorted(qualities_raw))
    extensions = tuple(sorted(data.get("extensions", {}).keys()))
    slash_bases = tuple(sorted(data.get("slash_bases", {}).keys()))

    # Tonalities: fixed set covering the 12 pitch classes × {major, minor}
    # plus common enharmonics found in the file names.
    pitch_classes = [
        "C", "C#", "Db", "D", "D#", "Eb", "E",
        "F", "F#", "Gb", "G", "G#", "Ab",
        "A", "A#", "Bb", "B",
    ]
    tonalities = tuple(
        f"{pc}_{mode}" for mode in ("major", "minor") for pc in pitch_classes
    )

    return L1Alphabet(
        roots=roots,
        qualities=qualities,
        extensions=extensions,
        slash_bases=slash_bases,
        tonalities=tonalities,
    )


# ---------------------------------------------------------------------------
# Sidecar parsing
# ---------------------------------------------------------------------------


@dataclass
class ChordEvent:
    """A single chord block in the L1 stream."""
    duration: str        # e.g. "4.0"
    root: str            # e.g. "C" or "vG#"
    quality: str         # e.g. "maj7" or "" for implicit major
    extensions: list[str] = field(default_factory=list)  # raw phrases, e.g. ["add 9"]
    slash_bass: str | None = None

    def to_tokens(self) -> list[str]:
        out = [".", f"DUR_{self.duration}", f"R_{self.root}"]
        out.append(EMPTY_QUALITY_TOKEN if self.quality == "" else f"Q_{self.quality}")
        out.extend(f"X_{e}" for e in self.extensions)
        if self.slash_bass is not None:
            out.extend(["/", f"R_{self.slash_bass}"])
        return out


@dataclass
class L1Parsed:
    """Structured result of parsing one sidecar."""
    header: list[str]              # STYLE_*, TONALITY_*, TYPE_* (emitted once)
    style_raw: str | None          # raw style string as found in the sidecar
    events: list["L1Event"]        # ordered events (structural + chord)

    @property
    def n_chords(self) -> int:
        return sum(1 for e in self.events if isinstance(e, ChordEvent))


@dataclass
class StructuralEvent:
    token: str  # one canonical token: "|", "|:", ":|", "FORM_A", etc.


L1Event = ChordEvent | StructuralEvent


# Filename pattern: <idx>_<songname>_<key>_<mode>_type_<type>.txt
# <key> may be like "C", "Db", "F#"; <mode> is "major" or "minor".
_FILENAME_TONALITY = re.compile(
    r"_([A-G](?:#|b)?)_(major|minor)_type_[^/]+\.txt$"
)


def extract_tonality_from_filename(path: str | Path) -> str | None:
    """Return e.g. 'C_major' or None if the filename doesn't match."""
    m = _FILENAME_TONALITY.search(str(path))
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def extract_type_from_path(path: str | Path) -> str | None:
    """Return e.g. '0_major' from '.../type_0_major/foo.txt'."""
    parts = Path(path).parts
    for p in parts:
        if p.startswith("type_"):
            return p[len("type_"):]
    return None


def _classify_form(raw: str) -> str | None:
    return FORM_RAW_TO_CANON.get(raw)


def parse_text_sidecar(
    path: str | Path,
    *,
    classify_style_fn=None,
) -> L1Parsed:
    """Parse a single ``.txt`` sidecar into structured L1 events.

    Parameters
    ----------
    path : str | Path
        Path to the sidecar file (``.txt`` containing a python-literal list).
    classify_style_fn : callable, optional
        Function mapping a raw iReal style string to a canonical STYLE label.
        If None, the raw style is used verbatim (prefixed ``STYLE_``).

    Returns
    -------
    L1Parsed
        Header tokens (emitted once at song start) and an ordered list of
        :class:`ChordEvent` and :class:`StructuralEvent` items.
    """
    p = Path(path)
    raw = ast.literal_eval(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Sidecar {p} did not parse to a list")

    # ----- Header: style + tonality + type -----
    idx = 0
    style_raw: str | None = None
    style_token: str | None = None
    if raw and raw[0] == "<style>":
        if len(raw) > 1:
            style_raw = raw[1]
            canonical = (
                classify_style_fn(style_raw) if classify_style_fn else style_raw
            )
            if canonical:
                style_token = f"STYLE_{canonical}"
        idx = 2

    tonality = extract_tonality_from_filename(p)
    type_label = extract_type_from_path(p)

    header: list[str] = []
    if style_token:
        header.extend(["<style>", style_token])
    if tonality:
        header.extend(["<tonality>", f"TONALITY_{tonality}"])
    if type_label:
        header.append(f"TYPE_{type_label}")

    # ----- Event stream -----
    events: list[L1Event] = []
    L = len(raw)
    i = idx
    while i < L:
        tok = raw[i]

        # Structural markers passed through unchanged
        if tok in ("|", "|:", ":|"):
            events.append(StructuralEvent(token=tok))
            i += 1
            continue

        # Form markers
        if tok in FORM_RAW_TO_CANON:
            events.append(StructuralEvent(token=FORM_RAW_TO_CANON[tok]))
            i += 1
            continue

        # Repeat_N / double bars / noise — skip silently
        if tok in _SIDECAR_REPEAT_RAW or tok in ("||", "b||", "e||"):
            i += 1
            continue

        # Chord event: starts with '.'
        if tok == ".":
            if i + 2 >= L:
                # Malformed trailing dot — stop
                break
            dur = _quantize_duration_str(raw[i + 1])
            root = str(raw[i + 2])
            quality = str(raw[i + 3]) if i + 3 < L else ""
            ext: list[str] = []
            k = i + 4
            while k < L:
                t = raw[k]
                if (
                    t == "."
                    or t == "/"
                    or t in ("|", "|:", ":|", "||", "b||", "e||")
                    or t in FORM_RAW_TO_CANON
                    or t in _SIDECAR_REPEAT_RAW
                ):
                    break
                ext.append(str(t))
                k += 1
            slash: str | None = None
            if k < L and raw[k] == "/":
                if k + 1 < L:
                    slash = str(raw[k + 1])
                    k += 2
                else:
                    k += 1  # dangling slash
            events.append(
                ChordEvent(
                    duration=dur,
                    root=root,
                    quality=quality,
                    extensions=ext,
                    slash_bass=slash,
                )
            )
            i = k
            continue

        # Unknown: skip with no noise (alphabet was scanned; this should be rare)
        i += 1

    return L1Parsed(header=header, style_raw=style_raw, events=events)


# ---------------------------------------------------------------------------
# Convenience: flatten an L1Parsed to a pure token stream
# ---------------------------------------------------------------------------


def parsed_to_tokens(parsed: L1Parsed) -> list[str]:
    """Flatten an :class:`L1Parsed` into a single list of L1 token strings."""
    out = list(parsed.header)
    for ev in parsed.events:
        if isinstance(ev, ChordEvent):
            out.extend(ev.to_tokens())
        else:
            out.append(ev.token)
    return out


def iter_chord_spans(tokens: Iterable[str]) -> list[tuple[int, int]]:
    """Given an L1 token list, return (start, end_exclusive) spans for each
    chord block (each starting with ``.`` and ending before the next chord
    start or structural marker). Useful for aligning with L2 chord blocks."""
    toks = list(tokens)
    spans: list[tuple[int, int]] = []
    i = 0
    L = len(toks)
    while i < L:
        if toks[i] == ".":
            start = i
            j = i + 1
            while j < L and toks[j] != "." and toks[j] not in (
                "|", "|:", ":|",
            ) and not toks[j].startswith("FORM_"):
                j += 1
            spans.append((start, j))
            i = j
        else:
            i += 1
    return spans


# ---------------------------------------------------------------------------
# L1 + L2 interleaver
# ---------------------------------------------------------------------------


def _split_l2_chord_blocks(l2_tokens: list[str]) -> list[list[str]]:
    """Split a flat L2 token list into per-chord blocks.

    Each block starts with ``CHORD_START`` and ends with ``CHORD_END``
    (inclusive). Tokens outside chord blocks (``BAR``, ``FORM_*``, header
    tokens, ``<start>``/``<end>``) are discarded — the L1 side owns all
    structural markers in the merged stream.
    """
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for t in l2_tokens:
        if t == "CHORD_START":
            cur = [t]
        elif t == "CHORD_END" and cur is not None:
            cur.append(t)
            blocks.append(cur)
            cur = None
        elif cur is not None:
            cur.append(t)
        # else: structural / header token — ignored
    return blocks


def merge_levels(
    parsed: L1Parsed,
    l2_tokens: list[str],
    *,
    start_token: str = "<start>",
    end_token: str = "<end>",
) -> list[str]:
    """Interleave L1 (symbolic) and L2 (MIDI/MPE) per chord.

    Invariant: ``parsed.n_chords`` must equal the number of ``CHORD_START``
    blocks in ``l2_tokens``. Both come from the same expanded song —
    mismatch means upstream generation is broken, so we raise loudly
    rather than silently drop data.

    Output stream:

        <start>
          header tokens  (STYLE_*, TONALITY_*, TYPE_*)
          # for each chord k:
          L1_chord_k  (. DUR_* R_* Q_* X_* ...)
          L2_chord_k  (CHORD_START DUR_* PV_* ... CHORD_END)
          # structural L1 markers (|, |:, :|, FORM_*) emitted between chord pairs
        <end>
    """
    l2_blocks = _split_l2_chord_blocks(l2_tokens)

    if parsed.n_chords != len(l2_blocks):
        raise ValueError(
            f"L1/L2 chord-count mismatch: L1 has {parsed.n_chords} chords, "
            f"L2 has {len(l2_blocks)} CHORD_START blocks. Upstream generator "
            f"is out of sync."
        )

    out: list[str] = [start_token]
    out.extend(parsed.header)

    k = 0  # index into l2_blocks
    for ev in parsed.events:
        if isinstance(ev, ChordEvent):
            out.extend(ev.to_tokens())     # L1 chord
            out.extend(l2_blocks[k])       # L2 chord (aligned)
            k += 1
        else:
            out.append(ev.token)           # structural (|, |:, :|, FORM_*)

    out.append(end_token)
    return out


# ---------------------------------------------------------------------------
# End-to-end convenience: paired .mid + .txt → merged token stream
# ---------------------------------------------------------------------------


def build_merged_tokens(
    midi_path: str | Path,
    text_path: str | Path,
    tokenizer,
) -> list[str]:
    """Parse one paired (``.mid``, ``.txt``) and return the merged token stream.

    Bypasses ``clean_chords`` (training-data sanitation designed for messy
    real MIDI) to preserve the L1↔L2 chord-count invariant on synthetic data.

    Parameters
    ----------
    midi_path, text_path : path-like
        The paired files. Must come from the same expanded song.
    tokenizer : MPETokenizer
        Used to produce L2 tokens via ``encode_chords`` and to canonicalize
        the L1 style label via ``classify_style``.

    Returns
    -------
    list[str]
        Merged L1+L2 token stream, ready for ``encode_to_ids``.
    """
    # L1
    parsed = parse_text_sidecar(
        text_path,
        classify_style_fn=getattr(tokenizer, "classify_style", None)
        or __import__("tokenizer").classify_style,
    )

    # L2 — parse without clean_chords so chord count matches L1 exactly
    from tokenizer import parse_mpe_midi  # local import to avoid cycle
    chords = parse_mpe_midi(str(midi_path))
    if not chords:
        raise ValueError(f"parse_mpe_midi returned no chords for {midi_path}")

    l2_tokens = tokenizer.encode_chords(
        chords,
        add_start_end=False,
        type_label=None,
        style_label=None,
        form_markers=None,
    )

    return merge_levels(parsed, l2_tokens)


# ---------------------------------------------------------------------------
# Merge + per-token chord spans (EigenSpace visibility rule B)
# ---------------------------------------------------------------------------


def merge_levels_with_spans(
    parsed: L1Parsed,
    l2_tokens: list[str],
    *,
    start_token: str = "<start>",
    end_token: str = "<end>",
) -> tuple[list[str], list[int]]:
    """Same output as :func:`merge_levels`, plus a per-token chord-index list.

    The spans implement **EigenSpace visibility rule B**:

        * Header tokens (``<start>``, ``STYLE_*``, ``TONALITY_*``, ``TYPE_*``)
          → ``-1``  (no chord visible yet; consumer emits defaults / zeros).
        * All tokens belonging to chord *k*'s L1 block + L2 block → ``k``.
        * Structural markers (``|``, ``|:``, ``:|``, ``FORM_*``) that sit
          between chord *k-1* and chord *k* → ``k-1``. The eigenspace of
          the **previous** chord persists across structural gaps — we never
          leak a future chord's eigenspace onto tokens that precede its
          first appearance.
        * ``<end>`` → last emitted chord index (or -1 if no chords).

    Returns
    -------
    (tokens, spans) : tuple[list[str], list[int]]
        ``len(tokens) == len(spans)``.
    """
    l2_blocks = _split_l2_chord_blocks(l2_tokens)

    if parsed.n_chords != len(l2_blocks):
        raise ValueError(
            f"L1/L2 chord-count mismatch: L1 has {parsed.n_chords} chords, "
            f"L2 has {len(l2_blocks)} CHORD_START blocks. Upstream generator "
            f"is out of sync."
        )

    tokens: list[str] = []
    spans:  list[int] = []

    # Header — no chord context yet.
    tokens.append(start_token); spans.append(-1)
    for h in parsed.header:
        tokens.append(h); spans.append(-1)

    k = 0           # next L2 block index
    last_k = -1     # last chord whose block was emitted (for structurals)

    for ev in parsed.events:
        if isinstance(ev, ChordEvent):
            for t in ev.to_tokens():          # L1 chord k
                tokens.append(t); spans.append(k)
            for t in l2_blocks[k]:            # L2 chord k
                tokens.append(t); spans.append(k)
            last_k = k
            k += 1
        else:
            # Structural marker — carry previous chord's eigenspace.
            tokens.append(ev.token); spans.append(last_k)

    tokens.append(end_token); spans.append(last_k)
    return tokens, spans


def build_merged_tokens_with_spans(
    midi_path: str | Path,
    text_path: str | Path,
    tokenizer,
) -> tuple[list[str], list[int]]:
    """Paired ``(.mid, .txt)`` → (merged tokens, per-token chord spans).

    See :func:`build_merged_tokens` for the parsing contract and
    :func:`merge_levels_with_spans` for the span semantics.
    """
    parsed = parse_text_sidecar(
        text_path,
        classify_style_fn=getattr(tokenizer, "classify_style", None)
        or __import__("tokenizer").classify_style,
    )

    from tokenizer import parse_mpe_midi
    chords = parse_mpe_midi(str(midi_path))
    if not chords:
        raise ValueError(f"parse_mpe_midi returned no chords for {midi_path}")

    l2_tokens = tokenizer.encode_chords(
        chords,
        add_start_end=False,
        type_label=None,
        style_label=None,
        form_markers=None,
    )

    return merge_levels_with_spans(parsed, l2_tokens)
# ---------------------------------------------------------------------------
# Merge + per-token chord spans (EigenSpace visibility rule B)
# ---------------------------------------------------------------------------


def merge_levels_with_spans(
    parsed: L1Parsed,
    l2_tokens: list[str],
    *,
    start_token: str = "<start>",
    end_token: str = "<end>",
) -> tuple[list[str], list[int]]:
    """Same output as :func:`merge_levels`, plus a per-token chord-index list.

    The spans implement **EigenSpace visibility rule B**:

        * Header tokens (``<start>``, ``STYLE_*``, ``TONALITY_*``, ``TYPE_*``)
          → ``-1``  (no chord visible yet; consumer emits defaults / zeros).
        * All tokens belonging to chord *k*'s L1 block + L2 block → ``k``.
        * Structural markers (``|``, ``|:``, ``:|``, ``FORM_*``) that sit
          between chord *k-1* and chord *k* → ``k-1``. The eigenspace of
          the **previous** chord persists across structural gaps — we never
          leak a future chord's eigenspace onto tokens that precede its
          first appearance.
        * ``<end>`` → last emitted chord index (or -1 if no chords).

    Returns
    -------
    (tokens, spans) : tuple[list[str], list[int]]
        ``len(tokens) == len(spans)``.
    """
    l2_blocks = _split_l2_chord_blocks(l2_tokens)

    if parsed.n_chords != len(l2_blocks):
        raise ValueError(
            f"L1/L2 chord-count mismatch: L1 has {parsed.n_chords} chords, "
            f"L2 has {len(l2_blocks)} CHORD_START blocks. Upstream generator "
            f"is out of sync."
        )

    tokens: list[str] = []
    spans:  list[int] = []

    # Header — no chord context yet.
    tokens.append(start_token); spans.append(-1)
    for h in parsed.header:
        tokens.append(h); spans.append(-1)

    k = 0           # next L2 block index
    last_k = -1     # last chord whose block was emitted (for structurals)

    for ev in parsed.events:
        if isinstance(ev, ChordEvent):
            for t in ev.to_tokens():          # L1 chord k
                tokens.append(t); spans.append(k)
            for t in l2_blocks[k]:            # L2 chord k
                tokens.append(t); spans.append(k)
            last_k = k
            k += 1
        else:
            # Structural marker — carry previous chord's eigenspace.
            tokens.append(ev.token); spans.append(last_k)

    tokens.append(end_token); spans.append(last_k)
    return tokens, spans


def build_merged_tokens_with_spans(
    midi_path: str | Path,
    text_path: str | Path,
    tokenizer,
) -> tuple[list[str], list[int]]:
    """Paired ``(.mid, .txt)`` → (merged tokens, per-token chord spans).

    See :func:`build_merged_tokens` for the parsing contract and
    :func:`merge_levels_with_spans` for the span semantics.
    """
    parsed = parse_text_sidecar(
        text_path,
        classify_style_fn=getattr(tokenizer, "classify_style", None)
        or __import__("tokenizer").classify_style,
    )

    from tokenizer import parse_mpe_midi
    chords = parse_mpe_midi(str(midi_path))
    if not chords:
        raise ValueError(f"parse_mpe_midi returned no chords for {midi_path}")

    l2_tokens = tokenizer.encode_chords(
        chords,
        add_start_end=False,
        type_label=None,
        style_label=None,
        form_markers=None,
    )

    return merge_levels_with_spans(parsed, l2_tokens)
