"""
build_parallel_dataset.py
=========================
Build the **parallel text-token dataset** that mirrors the 53-TET MPE MIDI
dataset one-to-one.

    Source MIDI tree (authoritative list of stems):
        dataset/midi_files/53_tet_mpe/type_<label>/<stem>.mid
    Output text tree (1:1 mirror, same stems):
        dataset/text_files/53_tet_mpe/type_<label>/<stem>.txt

Each `.txt` file contains a Python list (as `repr`) of tokens in the format
produced by notebook `01_musicXML_parser.ipynb` → `03_map_MIDI_to_FTT.ipynb`:

    ['<style>', 'Medium Swing', 'Form_A', '|',
     '.', '4.0', '^^F', 'supermajor-major',
     '.', '2.0', '^^F', 'supermajor-major',
     '.', '2.0', 'vvE', 'supraminor-supraminor-seventh',
     '|', ...]

i.e. the same cohesive 12-TET chord list (with roots / qualities already
normalised by `xmlTranslator.replaceTheseChords` + voicing-compatible names)
but remapped into 53-TET using the exact logic from notebook 03
(`build_chromatic_scale_53tet` + `NOTE_NAMES_53TET` + `convention.get_name`).

Pipeline
--------
  1. Walk every `*.mid` under `53_tet_mpe/type_<label>/`
  2. Parse the stem → (base_stem, key, mode, scale_type)
  3. Load the pre-existing 12-TET text at
       `dataset/text_files/12_tet_files/<base_stem>_<key>_<mode>.txt`
     (produced by notebook 01 after transposition to 12 tonalities).
  4. Remap chord roots + qualities from 12-TET to the 53-TET of
     `MODAL_SCALE_TYPES[scale_type]` with the tonic at `key`.
     Structural tokens (`.`, `|`, `|:`, `:|`, `/`, `Form_*`, `<style>`,
     `N.C.`, duration floats) are preserved verbatim.
  5. Write to `dataset/text_files/53_tet_mpe/type_<label>/<midi_stem>.txt`.

Usage
-----
    python build_parallel_dataset.py                 # full dataset
    python build_parallel_dataset.py --max-per-type 50  # smoke-test
    python build_parallel_dataset.py --dry-run       # count only, no writes
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Ensure src/ is on path for local imports
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import convention  # noqa: E402  — microtonal chord-name glossary


# =============================================================================
# MODAL SCALE CONFIGURATION (mirror of notebook 03)
# =============================================================================

NOTE_NAMES_53TET = [
    "C", "^C", "^^C", "vvC#", "vC#", "C#", "^C#", "^^C#", "vD", "D",
    "^D", "^^D", "vvD#", "vD#", "D#", "^^Eb", "vvE", "vE", "E", "^E",
    "^^E", "vF", "F", "^F", "^^F", "vvF#", "vF#", "F#", "^F#", "^^F#",
    "vG", "G", "^G", "^^G", "vvG#", "vG#", "G#", "^G#", "vvA", "vA",
    "A", "^A", "^^A", "vBb", "Bb", "^Bb", "^^Bb", "vvB", "vB", "B",
    "^B", "^^B", "vC",
]

MODAL_SCALE_TYPES = {
    'type_0_major':     {'hc_distances': [0, 9, 9, 4, 9, 9, 9]},
    'type_0_minor':     {'hc_distances': [0, 9, 4, 9, 9, 4, 9]},
    'type_1_neutral':   {'hc_distances': [0, 8, 7, 7, 9, 8, 7]},
    'type_1_minor':     {'hc_distances': [0, 7, 7, 8, 7, 7, 9]},
    'type_2_subminor':  {'hc_distances': [0, 9, 3, 10, 9, 3, 10]},
    'type_2_minor':     {'hc_distances': [0, 10, 9, 9, 3, 10, 9]},
    'type_3_major':     {'hc_distances': [0, 9, 8, 5, 9, 8, 5]},
    'type_3_minor':     {'hc_distances': [0, 5, 9, 9, 8, 5, 9]},
    'type_4_upmajor':   {'hc_distances': [0, 10, 9, 3, 9, 10, 9]},
    'type_4_minor':     {'hc_distances': [0, 9, 3, 10, 9, 3, 9]},
    'type_5_major_v2':  {'hc_distances': [0, 9, 9, 5, 8, 8, 9]},
    'type_5_minor':     {'hc_distances': [0, 9, 5, 9, 9, 5, 8]},
    'type_6_neutral_n': {'hc_distances': [0, 8, 7, 7, 9, 5, 10]},
    'type_6_minor':     {'hc_distances': [0, 10, 7, 8, 7, 7, 9]},
}

KEY_TO_POS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def build_chromatic_scale_53tet(hc_distances, tonic_position=0):
    """
    Build a 12-position chromatic scale in 53-TET, anchored to a tonic.

    Mirrors notebook 03's implementation: the 7 cumulative scale-tone steps are
    placed at their closest 12-TET positions, the 5 remaining chromatic
    positions are linearly interpolated, then the whole pattern is rotated so
    the tonic sits at `tonic_position`.
    """
    scale_7_steps = np.cumsum(hc_distances)
    relative_positions = [round(step * 12 / 53) for step in scale_7_steps]

    chromatic_from_root: List[Optional[int]] = [None] * 12
    for i, pos in enumerate(relative_positions):
        if pos < 12:
            chromatic_from_root[pos] = int(scale_7_steps[i])

    for i in range(12):
        if chromatic_from_root[i] is None:
            prev_pos, next_pos = -1, 12
            for j in range(i - 1, -1, -1):
                if chromatic_from_root[j] is not None:
                    prev_pos = j
                    break
            for j in range(i + 1, 12):
                if chromatic_from_root[j] is not None:
                    next_pos = j
                    break
            if prev_pos >= 0 and next_pos < 12:
                prev_step = chromatic_from_root[prev_pos]
                next_step = chromatic_from_root[next_pos]
                rng = next_step - prev_step
                positions = next_pos - prev_pos
                offset = i - prev_pos
                chromatic_from_root[i] = round(prev_step + (rng * offset / positions))
            elif prev_pos >= 0:
                prev_step = chromatic_from_root[prev_pos]
                rng = 53 - prev_step
                positions = 12 - prev_pos
                offset = i - prev_pos
                chromatic_from_root[i] = round(prev_step + (rng * offset / positions))
            else:
                chromatic_from_root[i] = round((i / 12) * 53)

    tonic_53tet = round(tonic_position * 53 / 12)
    return [
        (tonic_53tet + chromatic_from_root[(j - tonic_position) % 12]) % 53
        for j in range(12)
    ]


# =============================================================================
# CHORD-QUALITY INTERVAL MAP (mirror of notebook 03)
# =============================================================================
#
# These are the already-normalised quality tokens (voicing.natures + add/alter)
# produced by notebook 01 after `replaceTheseChords`.  We map each to its
# 12-TET interval recipe; the recipe is then converted to 53-TET step counts
# via the active `chromatic_scale` and `convention.get_name` gives the new
# microtonal name.

# Base (third, fifth, seventh) recipe per normalised nature.
# None means "no seventh" for triads.
NATURE_12TET_INTERVALS: Dict[str, Dict[str, Optional[int]]] = {
    # Majors
    'maj':     {'third': 4, 'fifth': 7, 'seventh': None},
    'maj6':    {'third': 4, 'fifth': 7, 'seventh': None},   # 6th is extension
    'maj7':    {'third': 4, 'fifth': 7, 'seventh': 11},
    # Minors
    'm':       {'third': 3, 'fifth': 7, 'seventh': None},
    'm6':      {'third': 3, 'fifth': 7, 'seventh': None},
    'm7':      {'third': 3, 'fifth': 7, 'seventh': 10},
    'm_maj7':  {'third': 3, 'fifth': 7, 'seventh': 11},
    # Dominants / sus
    'dom7':    {'third': 4, 'fifth': 7, 'seventh': 10},
    'sus':     {'third': 5, 'fifth': 7, 'seventh': None},
    'sus4':    {'third': 5, 'fifth': 7, 'seventh': None},
    'sus2':    {'third': 2, 'fifth': 7, 'seventh': None},
    'sus7':    {'third': 5, 'fifth': 7, 'seventh': 10},
    # Diminished / augmented
    'o':       {'third': 3, 'fifth': 6, 'seventh': None},
    'o7':      {'third': 3, 'fifth': 6, 'seventh': 9},
    'ø7':      {'third': 3, 'fifth': 6, 'seventh': 10},
    'o_maj7':  {'third': 3, 'fifth': 6, 'seventh': 11},
    'aug':     {'third': 4, 'fifth': 8, 'seventh': None},
    # Misc
    'power':   {'third': None, 'fifth': 7, 'seventh': None},
    'N.C.':    {'third': None, 'fifth': None, 'seventh': None},
}


# Tokens that are structural / non-chord — preserved verbatim.
STRUCTURAL_TOKENS = {
    '.', '|', '||', ':|', '|:', 'b||', 'e||', '/', 'N.C.', '<style>',
}


def _is_duration_token(t: str) -> bool:
    """True if the token looks like a duration float (e.g. '4.0')."""
    if not isinstance(t, str):
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


# =============================================================================
# 12-TET → 53-TET REMAP
# =============================================================================

def _step_from_12tet_interval(root_idx_12, root_step_53, interval_12, chromatic_scale):
    """
    Compute the 53-TET step count between a root and a note `interval_12`
    semitones above it, measured through the active chromatic_scale.
    """
    target_idx_12 = (root_idx_12 + interval_12) % 12
    target_step = chromatic_scale[target_idx_12]
    diff = target_step - root_step_53
    if diff < 0:
        diff += 53
    return diff


def _quality_supported(q: str) -> bool:
    """Known, directly-mappable quality root — ignores extensions like 'add 9'."""
    return q in NATURE_12TET_INTERVALS


def remap_tokens_to_53tet(tokens: List[str], scale_type: str, key: str) -> List[str]:
    """
    Apply the notebook-03 remap to a 12-TET cohesive token list.

    Inputs:
        tokens      — list like ['.', '4.0', 'G', 'maj7', '|', 'Form_A', '/', 'B', ...]
        scale_type  — 'type_0_major', etc.  (key from `MODAL_SCALE_TYPES`)
        key         — 'C', 'Db', ... (tonic pitch class in 12-TET)

    Output: same token list with roots remapped to 53-TET note names
    (NOTE_NAMES_53TET) and qualities remapped to microtonal names via
    `convention.get_name`.  Extensions (`add`/`alter`) and all structural
    tokens are preserved verbatim.  Slash-chord bass notes (the token
    immediately after '/') are remapped as plain root names (no quality).
    """
    if scale_type not in MODAL_SCALE_TYPES:
        raise ValueError(f"Unknown scale_type: {scale_type}")
    if key not in KEY_TO_POS:
        raise ValueError(f"Unknown key: {key}")

    chromatic_scale = build_chromatic_scale_53tet(
        MODAL_SCALE_TYPES[scale_type]['hc_distances'],
        tonic_position=KEY_TO_POS[key],
    )

    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        # Structural / non-chord tokens pass through.
        if (
            tok in STRUCTURAL_TOKENS
            or (isinstance(tok, str) and tok.startswith('Form_'))
            or _is_duration_token(tok)
        ):
            out.append(tok)
            # Slash-chord bass note: remap root only, skip quality parsing.
            if tok == '/' and i + 1 < n:
                bass = tokens[i + 1]
                if isinstance(bass, str) and bass in KEY_TO_POS:
                    bass_step = chromatic_scale[KEY_TO_POS[bass]]
                    out.append(NOTE_NAMES_53TET[bass_step % 53])
                    i += 2
                    continue
            i += 1
            continue

        # Root?  Must be a plain chromatic pitch-class like 'G', 'Bb', 'F#'.
        if isinstance(tok, str) and tok in KEY_TO_POS:
            root_text = tok
            # Peek next token — is it a known quality (nature)?
            quality = None
            has_quality_token = False
            if i + 1 < n:
                nxt = tokens[i + 1]
                if (
                    isinstance(nxt, str)
                    and nxt not in STRUCTURAL_TOKENS
                    and not nxt.startswith('Form_')
                    and not _is_duration_token(nxt)
                    and nxt not in KEY_TO_POS  # don't eat the next root
                ):
                    quality = nxt
                    has_quality_token = True

            intervals = NATURE_12TET_INTERVALS.get(
                quality,
                {'third': 4, 'fifth': 7, 'seventh': None},  # bare root defaults to major
            )

            root_idx_12 = KEY_TO_POS[root_text]
            root_step_53 = chromatic_scale[root_idx_12]

            steps_map = {}
            if intervals.get('third') is not None:
                steps_map['third'] = _step_from_12tet_interval(
                    root_idx_12, root_step_53, intervals['third'], chromatic_scale)
            else:
                steps_map['third'] = None
            if intervals.get('fifth') is not None:
                steps_map['fifth'] = _step_from_12tet_interval(
                    root_idx_12, root_step_53, intervals['fifth'], chromatic_scale)
            else:
                steps_map['fifth'] = None
            if intervals.get('seventh') is not None:
                steps_map['seventh'] = _step_from_12tet_interval(
                    root_idx_12, root_step_53, intervals['seventh'], chromatic_scale)
            else:
                steps_map['seventh'] = None

            q3 = convention.STEP_TO_SEMANTIC.get(steps_map['third']) if steps_map['third'] is not None else None
            q5 = convention.STEP_TO_SEMANTIC.get(steps_map['fifth']) if steps_map['fifth'] is not None else None
            q7 = convention.STEP_TO_SEMANTIC.get(steps_map['seventh']) if steps_map['seventh'] is not None else None

            new_quality = convention.get_name(q3, q5, q7)

            out.append(NOTE_NAMES_53TET[root_step_53 % 53])
            # Emit quality if the original had one, or if this root was bare
            # (in which case we fall back to whatever `convention.get_name`
            # produced for a plain major triad).
            if has_quality_token or new_quality:
                out.append(new_quality if new_quality else (quality or 'maj'))

            i += 2 if has_quality_token else 1
            continue

        # Anything else (extension token like 'add 9', 'alter b5', 'N.C.',
        # leftover legacy text) — pass through unchanged.
        out.append(tok)
        i += 1

    return out


# =============================================================================
# FILENAME PARSING
# =============================================================================

_STEM_REGEX = re.compile(
    r'^(?P<idx>\d+)_(?P<song>.+?)_(?P<key>[A-G][#b]?)_(?P<mode>major|minor)_(?P<stype>type_\d+_[A-Za-z0-9_]+)$'
)


def parse_midi_stem(stem: str) -> Optional[Dict[str, str]]:
    """
    Parse a 53-TET MPE MIDI stem into its components.

    Example:
        '03049_O Bebado e a Equilibrista BCarvalho_Db_major_type_0_major'
    →
        {'idx': '03049',
         'song': 'O Bebado e a Equilibrista BCarvalho',
         'key': 'Db', 'mode': 'major',
         'stype': 'type_0_major',
         'base_stem': '03049_O Bebado e a Equilibrista BCarvalho_Db_major'}
    """
    m = _STEM_REGEX.match(stem)
    if m is None:
        return None
    d = m.groupdict()
    d['base_stem'] = f"{d['idx']}_{d['song']}_{d['key']}_{d['mode']}"
    return d


# =============================================================================
# CORE REBUILD LOOP
# =============================================================================

def _load_12tet_tokens(txt_path: Path) -> Optional[List[str]]:
    """
    Parse a 12-TET text file (written by notebook 01) into a token list.
    Returns None on failure.
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        toks = ast.literal_eval(content)
        if not isinstance(toks, list):
            return None
        return [str(t) for t in toks]
    except Exception:
        return None


def build_parallel_dataset(
    midi_root: Path,
    text12_dir: Path,
    out_root: Path,
    max_per_type: Optional[int] = None,
    dry_run: bool = False,
) -> Dict:
    """
    Rebuild the 53-TET text mirror from the pre-existing 12-TET text files.

    Returns a stats dict.
    """
    type_dirs = sorted(
        d for d in midi_root.iterdir()
        if d.is_dir() and d.name.startswith('type_') and not d.name.endswith('_file')
    )
    if not type_dirs:
        raise RuntimeError(f"No type_<label>/ folders under {midi_root}")

    total_midi = total_written = 0
    total_missing_text = total_bad_stem = total_bad_parse = 0
    missing_sample: List[str] = []
    stype_counts: Dict[str, int] = {}

    t0 = time.time()
    for td in type_dirs:
        files = sorted(td.glob('*.mid'))
        if max_per_type is not None:
            files = files[:max_per_type]
        out_dir = out_root / td.name
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        n_written_this_type = 0
        for midi_path in tqdm(files, desc=td.name, unit='file', leave=False):
            total_midi += 1
            parsed = parse_midi_stem(midi_path.stem)
            if parsed is None:
                total_bad_stem += 1
                continue

            src_txt = text12_dir / f"{parsed['base_stem']}.txt"
            if not src_txt.exists():
                total_missing_text += 1
                if len(missing_sample) < 5:
                    missing_sample.append(str(src_txt))
                continue

            tokens12 = _load_12tet_tokens(src_txt)
            if tokens12 is None:
                total_bad_parse += 1
                continue

            try:
                tokens53 = remap_tokens_to_53tet(
                    tokens12, scale_type=parsed['stype'], key=parsed['key']
                )
            except Exception:
                total_bad_parse += 1
                continue

            if not dry_run:
                out_path = out_dir / f"{midi_path.stem}.txt"
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(repr(tokens53))
            total_written += 1
            n_written_this_type += 1

        stype_counts[td.name] = n_written_this_type

    dt = time.time() - t0
    return {
        'total_midi':          total_midi,
        'total_written':       total_written,
        'total_missing_text':  total_missing_text,
        'total_bad_stem':      total_bad_stem,
        'total_bad_parse':     total_bad_parse,
        'missing_sample':      missing_sample,
        'per_type_counts':     stype_counts,
        'elapsed_sec':         dt,
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--midi-dir',  default=str(_ROOT_DIR / 'dataset/midi_files/53_tet_mpe'),
                    help='Root 53-TET MPE MIDI folder (contains type_* subdirs)')
    ap.add_argument('--text12-dir', default=str(_ROOT_DIR / 'dataset/text_files/12_tet_files'),
                    help='Source 12-TET text files (flat folder, per-song×transposition)')
    ap.add_argument('--out-dir',   default=str(_ROOT_DIR / 'dataset/text_files/53_tet_mpe'),
                    help='Output mirror: text_files/53_tet_mpe/type_<label>/<stem>.txt')
    ap.add_argument('--max-per-type', type=int, default=None,
                    help='Smoke-test: only process N files per type folder')
    ap.add_argument('--dry-run', action='store_true',
                    help='Scan + remap, but do not write output')
    args = ap.parse_args()

    midi_root  = Path(args.midi_dir).resolve()
    text12_dir = Path(args.text12_dir).resolve()
    out_root   = Path(args.out_dir).resolve()

    print(f"MIDI source:   {midi_root}")
    print(f"12-TET source: {text12_dir}")
    print(f"Output mirror: {out_root}")
    if args.max_per_type is not None:
        print(f"  MAX-PER-TYPE: {args.max_per_type}")
    if args.dry_run:
        print("  DRY RUN: no files will be written")

    stats = build_parallel_dataset(
        midi_root=midi_root,
        text12_dir=text12_dir,
        out_root=out_root,
        max_per_type=args.max_per_type,
        dry_run=args.dry_run,
    )

    print("\n=== SUMMARY ===")
    print(f"midi scanned     : {stats['total_midi']:,}")
    print(f"text written     : {stats['total_written']:,}")
    print(f"missing 12-TET   : {stats['total_missing_text']:,}")
    print(f"bad MIDI stems   : {stats['total_bad_stem']:,}")
    print(f"bad parses       : {stats['total_bad_parse']:,}")
    print(f"elapsed          : {stats['elapsed_sec']:.1f} s")
    if stats['missing_sample']:
        print("  sample missing text files:")
        for s in stats['missing_sample']:
            print("   ", s)
    if stats['per_type_counts']:
        print("\n  per-type written:")
        for k, v in stats['per_type_counts'].items():
            print(f"    {k}: {v:,}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
