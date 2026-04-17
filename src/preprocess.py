"""
preprocess.py
=============
Pre-processing pipeline for dual-channel training data.

Produces aligned data for both channels of the architecture:
  Channel 1 — MIDI tokens   → attention mechanism (sequential patterns)
  Channel 2 — EigenSpace 4D → spatial embedding    (harmonic geometry)

Data Format (per song)
----------------------
  {
    "file":           "song.mid",
    "token_ids":      [int, ...],           # flat MIDI token sequence
    "token_strs":     [str, ...],           # token strings (for debugging)
    "eigenspace_4d":  [[α, β, γ, δ], ...],   # one 4D vector per chord
    "chord_spans":    [int, ...],           # maps each token → its chord index
    "n_chords":       int,                  # number of chords in the song
    "n_tokens":       int,                  # length of the token sequence
  }

Key invariant:
  For every token position i:
    chord_idx = chord_spans[i]
    if chord_idx >= 0:
      eigenspace_4d[chord_idx] → the (α, β, γ, δ) for that token
    else:
      chord_idx == -1 → non-chord token (<start>, <end>, BAR)
          → use default embedding (1.0, 1.0, 2.0, 1.0)

Usage
-----
  from preprocess import preprocess_song, preprocess_dataset

  # Single song
  result = preprocess_song("path/to/song.mid")

  # Full dataset
  preprocess_dataset("dataset/midi_files/53_tet_mpe", "dataset/preprocessed")

  # For generation (on-the-fly from token strings)
  from preprocess import tokens_to_eigenspace_online
  eigen_4d, spans = tokens_to_eigenspace_online(token_strs, computer)
"""

import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent to path for imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Local imports
from eigenspace import (
    EigenSpaceComputer, classify_intervals,
    get_53tet_ratio, DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA,
    N_EIGEN, TET_53,
)

# Tokenizer import
from tokenizer import (
    MPETokenizer, parse_mpe_midi, clean_chords,
    CHORD_START_TOKEN, CHORD_END_TOKEN,
    TYPE_PREFIX, TYPE_LABELS,
    STYLE_PREFIX, STYLE_LABELS,
    FORM_PREFIX, FORM_LABELS,
    _extract_type_label,
)


# -----------------------------------------------------------------------------
# PARALLEL-JSON SIDECAR (form + style metadata from iRealXML)
# -----------------------------------------------------------------------------

def _parallel_sidecar_path(midi_path: Path) -> Optional[Path]:
    """
    Map a 53-TET MPE MIDI path to its parallel JSON sidecar produced by
    src/build_parallel_dataset.py.

        dataset/midi_files/53_tet_mpe/type_<label>/<stem>.mid
      → dataset/midi_files/53_tet_mpe/type_<label>_file/<stem>.json
    """
    parent = midi_path.parent
    if not parent.name.startswith("type_"):
        return None
    sidecar_dir = parent.with_name(parent.name + "_file")
    return sidecar_dir / (midi_path.stem + ".json")


def _extract_form_markers_from_chord_tokens(chord_tokens: List) -> Dict[int, str]:
    """
    Scan an iRealXML-derived `chord_tokens` stream (as stored in the
    parallel JSON sidecar) and return {bar_0idx: raw_form_label}.

    Bar boundaries are '|' tokens. A Form_* marker that appears between
    bar-pipe N and bar-pipe N+1 belongs to bar N+1 (0-indexed), i.e. the
    bar that the marker introduces.
    """
    markers: Dict[int, str] = {}
    bar_0idx = -1  # -1 = before the first bar
    for tok in chord_tokens:
        if tok == '|':
            bar_0idx += 1
        elif isinstance(tok, str) and tok.startswith('Form_'):
            markers[bar_0idx + 1] = tok
    return markers


def load_parallel_metadata(midi_path: str) -> Dict:
    """
    Load the parallel-dataset JSON sidecar for a MIDI file, if present.

    Returns:
        {'form_markers': {bar_0idx: 'Form_X'},
         'style_canonical': str | None,
         'style_raw':       str | None}
        Missing/invalid sidecar → empty defaults.
    """
    sidecar = _parallel_sidecar_path(Path(midi_path))
    if sidecar is None or not sidecar.exists():
        return {'form_markers': {}, 'style_canonical': None, 'style_raw': None}
    try:
        with open(sidecar, 'r') as f:
            meta = json.load(f)
    except Exception:
        return {'form_markers': {}, 'style_canonical': None, 'style_raw': None}
    chord_tokens = meta.get('chord_tokens') or []
    return {
        'form_markers':   _extract_form_markers_from_chord_tokens(chord_tokens),
        'style_canonical': meta.get('style_canonical'),
        'style_raw':       meta.get('style_raw'),
    }


# Lazy torch
_torch_module = None
def _torch():
    global _torch_module
    if _torch_module is None:
        import torch
        _torch_module = torch
    return _torch_module


# =============================================================================
# CHORD-LEVEL EIGENSPACE COMPUTATION
# =============================================================================

def _chord_to_eigenspace(
    chord: dict,
    **_kwargs,
) -> List[float]:
    """
    Compute the 4D EigenSpace vector for a single chord event.
    
    Takes a chord dict (from parse_mpe_midi) and returns [α, β, γ, δ].
    Dissonance is root-dependent — δ captures the root pitch class.
    
    Args:
        chord: dict with 'notes' → list of {'step_53': int, 'velocity': int}
        
    Returns:
        [alpha, beta, gamma, delta] — 4 floats
    """
    pitches = sorted(set(n['step_53'] for n in chord['notes']))
    
    if len(pitches) < 2:
        return [DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA]
    
    # Fold to one octave (mod 53) relative to root
    root = min(pitches)
    root_pc = root % TET_53
    intervals = sorted(set((p - root) % TET_53 for p in pitches))
    
    # Classify → (α, β, γ, δ)
    alpha, beta, gamma, delta = classify_intervals(intervals, root_pc=root_pc)
    
    return [float(alpha), float(beta), float(gamma), float(delta)]


# =============================================================================
# CHORD-SPAN INDEX
# =============================================================================

def _build_chord_spans(token_strs: List[str]) -> List[int]:
    """
    Map each token position to its parent chord index.
    
    Returns a list of length len(token_strs) where:
      spans[i] = chord_index (0, 1, 2, ...)  if token i is inside a chord
      spans[i] = -1                           if token i is non-chord (BAR, <start>, etc.)
    
    A chord includes everything from CHORD_START through CHORD_END (inclusive).
    
    Args:
        token_strs: Flat list of token strings
        
    Returns:
        List[int] of same length as token_strs
    """
    spans = [-1] * len(token_strs)
    chord_idx = -1
    inside_chord = False
    
    for i, tok in enumerate(token_strs):
        if tok == CHORD_START_TOKEN:
            chord_idx += 1
            inside_chord = True
            spans[i] = chord_idx
        elif tok == CHORD_END_TOKEN:
            spans[i] = chord_idx
            inside_chord = False
        elif inside_chord:
            spans[i] = chord_idx
    
    return spans


# =============================================================================
# TYPE LABEL EXTRACTION
# =============================================================================

def extract_type_label(midi_path: str) -> Optional[str]:
    """
    Extract the transformation type label from a MIDI file path.
    Delegates to tokenizer._extract_type_label.
    """
    return _extract_type_label(midi_path)


# =============================================================================
# SINGLE-SONG PREPROCESSING
# =============================================================================

def preprocess_song(
    midi_path: str,
    tokenizer: Optional[MPETokenizer] = None,
    speed: float = 1.0,
    type_label: Optional[str] = None,
    style_label: Optional[str] = None,
    form_markers: Optional[Dict[int, str]] = None,
    use_parallel_metadata: bool = True,
    **_kwargs,
) -> Optional[Dict]:
    """
    Full dual-channel pre-processing for one MIDI file.
    
    Pipeline:
      1. Parse MIDI → chord events (onset, duration, notes)
      2. Extract type label from path (if not provided)
      2b. Load parallel JSON sidecar → style_canonical + FORM markers
          (only if not explicitly overridden via args)
      3. Tokenize → flat token_ids with TYPE + STYLE + FORM conditioning
      4. For each chord: compute EigenSpace 4D (spatial channel)
      5. Build chord_spans (token position → chord index)
      6. Return aligned dict
    
    Args:
        midi_path: Path to 53-TET MPE MIDI file
        tokenizer: MPETokenizer instance (creates default if None)
        speed: Playback speed multiplier
        type_label: Transformation type (e.g. "0_major"). Auto-detected if None.
        style_label: Musical style (e.g. "jazz", "blues"). If None and
                     use_parallel_metadata is True, loaded from sidecar.
        form_markers: Optional explicit {bar_0idx: raw_form} dict. If None and
                      use_parallel_metadata is True, loaded from sidecar.
        use_parallel_metadata: When True, read the matching
                               type_<label>_file/<stem>.json sidecar for
                               style + form metadata.
        
    Returns:
        Dict with all channels aligned, or None on failure
    """
    if tokenizer is None:
        tokenizer = MPETokenizer()
    
    midi_path = Path(midi_path)
    
    # Step 1: Parse MIDI → chord events, then clean
    chords = parse_mpe_midi(str(midi_path), speed=speed)
    if not chords:
        return None
    chords = clean_chords(chords)
    
    # Step 2: Extract type label from path if not provided
    if type_label is None:
        type_label = extract_type_label(str(midi_path))

    # Step 2b: Pull style + form metadata from the parallel JSON sidecar
    # (built by src/build_parallel_dataset.py from the iRealXML source).
    if use_parallel_metadata and (style_label is None or form_markers is None):
        meta = load_parallel_metadata(str(midi_path))
        if style_label is None:
            style_label = meta.get('style_canonical')
        if form_markers is None:
            form_markers = meta.get('form_markers') or None

    # Step 3: Tokenize (MIDI channel) — with TYPE + STYLE + FORM conditioning
    token_strs = tokenizer.encode_chords(chords, add_start_end=True,
                                          type_label=type_label,
                                          style_label=style_label,
                                          form_markers=form_markers)
    token_ids = tokenizer.encode_to_ids(token_strs)
    
    # Step 4: EigenSpace 4D per chord — (α, β, γ, δ) position in harmonic space
    eigenspace_4d = []
    for chord in chords:
        vec = _chord_to_eigenspace(chord)
        eigenspace_4d.append(vec)
    
    # Step 5: Chord-span index
    chord_spans = _build_chord_spans(token_strs)
    
    # Validation: n_chords should match
    n_chords_from_spans = max(chord_spans) + 1 if chord_spans else 0
    n_chords_from_eigen = len(eigenspace_4d)
    assert n_chords_from_spans == n_chords_from_eigen, (
        f"Chord count mismatch: spans say {n_chords_from_spans}, "
        f"eigenspace has {n_chords_from_eigen}"
    )
    
    return {
        'file': midi_path.name,
        'type_label': type_label,
        'style_label': style_label,
        'token_ids': token_ids,
        'token_strs': token_strs,
        'eigenspace_4d': eigenspace_4d,
        'chord_spans': chord_spans,
        'n_chords': n_chords_from_eigen,
        'n_tokens': len(token_ids),
    }


# =============================================================================
# ON-THE-FLY EIGENSPACE (for generation / inference)
# =============================================================================

def tokens_to_eigenspace_online(
    token_strs: List[str],
    **_kwargs,
) -> Tuple[List[List[float]], List[int]]:
    """
    Compute EigenSpace 4D from generated token strings (on-the-fly).
    
    Used during GENERATION (autoregressive feedback):
    The model generates MIDI tokens → we decode pitches from each chord →
    compute EigenSpace → feed back as spatial embedding for the next step.
    
    Handles both PV_<step>_<vel> (new) and P_<step> (legacy) token formats.
    
    Args:
        token_strs: Token string sequence (possibly partial / in-progress)
        
    Returns:
        (eigenspace_4d, chord_spans):
          eigenspace_4d: list of [α, β, γ, δ] per chord
          chord_spans: list of int, one per token position
    """
    chord_spans = []
    eigenspace_4d = []
    chord_idx = -1
    inside_chord = False
    current_pitches = []
    
    for i, tok in enumerate(token_strs):
        if tok == CHORD_START_TOKEN:
            chord_idx += 1
            inside_chord = True
            current_pitches = []
            chord_spans.append(chord_idx)
            
        elif tok == CHORD_END_TOKEN:
            # Chord complete — compute its EigenSpace
            if current_pitches:
                root = min(current_pitches)
                root_pc = root % TET_53
                intervals = sorted(set((p - root) % TET_53 for p in current_pitches))
                alpha, beta, gamma, delta = classify_intervals(intervals, root_pc=root_pc)
                eigenspace_4d.append([float(alpha), float(beta), float(gamma), float(delta)])
            else:
                eigenspace_4d.append([DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA])
            
            chord_spans.append(chord_idx)
            inside_chord = False
            
        elif inside_chord:
            # Collect pitches from PV_ or P_ tokens
            if tok.startswith("PV_"):
                current_pitches.append(int(tok.split("_")[1]))
            elif tok.startswith("P_"):
                current_pitches.append(int(tok[2:]))
            chord_spans.append(chord_idx)
            
        else:
            chord_spans.append(-1)
    
    return eigenspace_4d, chord_spans


def expand_eigenspace_to_tokens(
    eigenspace_4d: List[List[float]],
    chord_spans: List[int],
) -> np.ndarray:
    """
    Expand per-chord EigenSpace vectors to per-token vectors.
    
    Given:
      eigenspace_4d: (n_chords, 4) — one vector per chord
      chord_spans:   (n_tokens,)   — maps each token to its chord index
      
    Returns:
      np.ndarray of shape (n_tokens, 4) — one vector per token position
      Non-chord tokens get [DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA]
    """
    n_tokens = len(chord_spans)
    default = [DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA]
    result = np.tile(default, (n_tokens, 1)).astype(np.float32)
    
    eigen_arr = np.array(eigenspace_4d, dtype=np.float32)
    
    for i, chord_idx in enumerate(chord_spans):
        if chord_idx >= 0 and chord_idx < len(eigen_arr):
            result[i] = eigen_arr[chord_idx]
    
    return result


# =============================================================================
# DATASET PREPROCESSING
# =============================================================================

def preprocess_dataset(
    midi_dir: str,
    output_dir: str,
    tokenizer: Optional[MPETokenizer] = None,
    max_files: Optional[int] = None,
    speed: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """
    Batch pre-process all MIDI files into dual-channel training data.
    
    Walks subdirectories matching type_* pattern to extract transformation
    type labels, or processes flat directories with type auto-detection.
    
    Saves:
      <output_dir>/vocab.json              — tokenizer vocabulary
      <output_dir>/eigenspace_stats.json   — EigenSpace distribution statistics
      <output_dir>/dataset_stats.json      — dataset-level statistics
      <output_dir>/songs/*.json            — per-song dual-channel data
    
    Args:
        midi_dir: Directory containing 53-TET MPE MIDI files (with type_* subdirs)
        output_dir: Output directory
        tokenizer: MPETokenizer (creates default if None)
        max_files: Limit number of files (None = all)
        speed: Speed multiplier
        verbose: Print progress
        
    Returns:
        Dict of statistics
    """
    if tokenizer is None:
        tokenizer = MPETokenizer()
    
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect MIDI files — walk type_* subdirectories if they exist
    files = []
    type_subdirs = sorted([d for d in midi_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("type_")])
    
    if type_subdirs:
        # Structured dataset with type subdirectories
        for subdir in type_subdirs:
            subfiles = sorted(subdir.glob("*.mid"))
            files.extend(subfiles)
        if verbose:
            print(f"Found {len(type_subdirs)} type subdirectories:")
            for sd in type_subdirs:
                n = len(list(sd.glob("*.mid")))
                print(f"  {sd.name}: {n:,} files")
    else:
        # Flat directory — type will be extracted from filename
        files = sorted(midi_dir.glob("*.mid"))
    
    if max_files:
        files = files[:max_files]
    
    if verbose:
        print(f"Pre-processing {len(files):,} files from {midi_dir}/")
    
    # Process each file
    results = []
    failed = 0
    all_eigenspace = []   # collect all chord vectors for global stats
    total_chords = 0
    total_tokens = 0
    type_counts = {}  # track count per type
    
    for i, f in enumerate(files):
        try:
            result = preprocess_song(
                str(f), tokenizer=tokenizer, speed=speed,
            )
            if result:
                compact = {
                    'file': result['file'],
                    'type_label': result.get('type_label'),
                    'token_ids': result['token_ids'],
                    'eigenspace_4d': result['eigenspace_4d'],
                    'chord_spans': result['chord_spans'],
                    'n_chords': result['n_chords'],
                    'n_tokens': result['n_tokens'],
                }
                results.append(compact)
                all_eigenspace.extend(result['eigenspace_4d'])
                total_chords += result['n_chords']
                total_tokens += result['n_tokens']
                # Track type distribution
                tl = result.get('type_label', 'unknown')
                type_counts[tl] = type_counts.get(tl, 0) + 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if verbose and failed <= 5:
                print(f"  Error: {f.name}: {e}")
        
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(files)} ({failed} failed)")
    
    if verbose:
        print(f"\nDone: {len(results)} songs, {failed} failures")
        print(f"  Total chords: {total_chords}")
        print(f"  Total tokens: {total_tokens}")
    
    # EigenSpace statistics (4D: α, β, γ, δ)
    if all_eigenspace:
        eigen_arr = np.array(all_eigenspace, dtype=np.float32)
        eigen_stats = {
            'n_chords': len(all_eigenspace),
            'alpha_mean': float(np.mean(eigen_arr[:, 0])),
            'alpha_std':  float(np.std(eigen_arr[:, 0])),
            'beta_mean':  float(np.mean(eigen_arr[:, 1])),
            'beta_std':   float(np.std(eigen_arr[:, 1])),
            'gamma_mean': float(np.mean(eigen_arr[:, 2])),
            'gamma_std':  float(np.std(eigen_arr[:, 2])),
            'delta_mean': float(np.mean(eigen_arr[:, 3])),
            'delta_std':  float(np.std(eigen_arr[:, 3])),
        }
    else:
        eigen_stats = {}
    
    dataset_stats = {
        'total_files': len(files),
        'successful': len(results),
        'failed': failed,
        'total_tokens': total_tokens,
        'total_chords': total_chords,
        'vocab_size': tokenizer.vocab_size,
        'avg_tokens_per_song': round(total_tokens / max(1, len(results)), 1),
        'avg_chords_per_song': round(total_chords / max(1, len(results)), 1),
        'eigenspace_stats': eigen_stats,
        'type_distribution': type_counts,
    }
    
    # Save outputs
    if verbose:
        print(f"\nSaving to {output_dir}/")
    
    tokenizer.save_vocab(str(output_dir / "vocab.json"))
    
    with open(output_dir / "eigenspace_stats.json", 'w') as f:
        json.dump(eigen_stats, f, indent=2)
    
    with open(output_dir / "dataset_stats.json", 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Save dual-channel data — one JSON per song for streaming / random access
    data_dir = output_dir / "songs"
    data_dir.mkdir(exist_ok=True)
    
    for result in results:
        song_name = Path(result['file']).stem
        song_path = data_dir / f"{song_name}.json"
        with open(song_path, 'w') as f:
            json.dump(result, f)
    
    # Also save an index file
    index = [{'file': r['file'], 'n_tokens': r['n_tokens'], 'n_chords': r['n_chords']}
             for r in results]
    with open(output_dir / "index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    if verbose:
        print(f"  vocab.json ({tokenizer.vocab_size} tokens)")
        print(f"  songs/ ({len(results)} files)")
        print(f"  index.json")
        print(f"  eigenspace_stats.json")
        print(f"  dataset_stats.json")
        if eigen_stats:
            print(f"\n  EigenSpace distribution:")
            print(f"    α: {eigen_stats['alpha_mean']:.4f} ± {eigen_stats['alpha_std']:.4f}")
            print(f"    β: {eigen_stats['beta_mean']:.4f} ± {eigen_stats['beta_std']:.4f}")
            print(f"    γ: {eigen_stats['gamma_mean']:.4f} ± {eigen_stats['gamma_std']:.4f}")
            print(f"    δ: {eigen_stats['delta_mean']:.4f} ± {eigen_stats['delta_std']:.4f}")
        if type_counts:
            print(f"\n  Type distribution:")
            for tl, count in sorted(type_counts.items()):
                print(f"    {tl}: {count:,}")
    
    return dataset_stats


# =============================================================================
# PYTORCH DATASET (dual-channel)
# =============================================================================

class DualChannelDataset:
    """
    PyTorch-compatible dataset for dual-channel training.
    
    Each item returns (token_ids, eigenspace_4d, attention_mask):
      token_ids:     (block_size,) int64   — input for token embedding → attention
      eigenspace_4d: (block_size, 4) float32 — input for spatial embedding
      target_ids:    (block_size,) int64   — shifted target for next-token loss
    
    The model combines these as:
      h = tok_emb(token_ids) + pos_emb + eigen_emb(eigenspace_4d)
    
    Usage:
        dataset = DualChannelDataset(
            data_dir="dataset/preprocessed/songs",
            tokenizer=MPETokenizer(),
            block_size=512,
        )
    """
    
    def __init__(self, data_dir: str, tokenizer: MPETokenizer,
                 block_size: int = 512, max_files: Optional[int] = None,
                 verbose: bool = True):
        """
        Load pre-processed dual-channel data.
        
        Args:
            data_dir: Directory with per-song .json files (from preprocess_dataset)
            tokenizer: MPETokenizer instance (for pad_id and vocab_size)
            block_size: Context window size
            max_files: Limit number of files
            verbose: Print progress
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.token_to_id["<pad>"]
        
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.json"))
        if max_files:
            files = files[:max_files]
        
        if verbose:
            print(f"Loading {len(files)} pre-processed songs from {data_dir}/")
        
        self.songs = []
        for f in files:
            with open(f, 'r') as fp:
                song = json.load(fp)
            self.songs.append(song)
        
        if verbose:
            n_tokens = sum(s['n_tokens'] for s in self.songs)
            n_chords = sum(s['n_chords'] for s in self.songs)
            print(f"  Loaded: {len(self.songs)} songs, {n_tokens} tokens, {n_chords} chords")
    
    def __len__(self):
        return len(self.songs)
    
    def __getitem__(self, idx):
        """
        Returns (x_tokens, x_eigen, y_tokens):
          x_tokens: (block_size,) int64
          x_eigen:  (block_size, 4) float32
          y_tokens: (block_size,) int64
        """
        torch = _torch()
        song = self.songs[idx]
        
        token_ids = song['token_ids']
        eigenspace_4d = song.get('eigenspace_4d', song.get('eigenspace_3d', []))
        chord_spans = song['chord_spans']
        
        # Pad to block_size + 1 (need one extra for shift)
        total_len = self.block_size + 1
        
        if len(token_ids) >= total_len:
            ids = token_ids[:total_len]
            spans = chord_spans[:total_len]
        else:
            ids = token_ids + [self.pad_id] * (total_len - len(token_ids))
            spans = chord_spans + [-1] * (total_len - len(chord_spans))
        
        # Expand eigenspace to per-token (4D: α, β, γ, δ)
        default = [DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA]
        eigen_expanded = np.tile(default, (total_len, 1)).astype(np.float32)
        eigen_arr = np.array(eigenspace_4d, dtype=np.float32)
        # Handle legacy 3D data: pad with default δ=1.0
        if eigen_arr.ndim == 2 and eigen_arr.shape[1] == 3:
            delta_col = np.full((eigen_arr.shape[0], 1), DEFAULT_DELTA, dtype=np.float32)
            eigen_arr = np.concatenate([eigen_arr, delta_col], axis=1)
        
        for i, chord_idx in enumerate(spans):
            if 0 <= chord_idx < len(eigen_arr):
                eigen_expanded[i] = eigen_arr[chord_idx]
        
        # Split into input/target
        ids_t = torch.tensor(ids, dtype=torch.long)
        eigen_t = torch.tensor(eigen_expanded, dtype=torch.float32)
        
        x_tokens = ids_t[:-1]       # (block_size,)
        x_eigen  = eigen_t[:-1]     # (block_size, 4)
        y_tokens = ids_t[1:]        # (block_size,)
        
        return x_tokens, x_eigen, y_tokens


# =============================================================================
# VALIDATION
# =============================================================================

def validate_song(result: Dict, verbose: bool = True) -> Dict:
    """
    Validate a pre-processed song for consistency.
    
    Checks:
      1. token_ids and chord_spans have same length
      2. chord_spans max index == n_chords - 1
      3. eigenspace_4d has n_chords entries
      4. All α ≤ β ≤ γ (tetrahedron constraint)
      5. No chord index gaps (monotonically increasing)
    
    Returns:
        Dict of checks passed/failed
    """
    checks = {}
    
    # 1. Length match
    checks['length_match'] = len(result['token_ids']) == len(result['chord_spans'])
    
    # 2. Chord count from spans
    max_span = max(result['chord_spans']) if result['chord_spans'] else -1
    checks['span_count_match'] = (max_span + 1) == result['n_chords']
    
    # 3. EigenSpace count
    eigen_key = 'eigenspace_4d' if 'eigenspace_4d' in result else 'eigenspace_3d'
    checks['eigen_count_match'] = len(result[eigen_key]) == result['n_chords']
    
    # 4. Tetrahedron constraint
    violations = 0
    for vec in result[eigen_key]:
        a, b, g = vec[0], vec[1], vec[2]
        if not (a <= b + 1e-6 and b <= g + 1e-6):
            violations += 1
    checks['tetrahedron_ok'] = violations == 0
    
    # 5. Monotonic chord indices
    prev = -1
    monotonic = True
    for s in result['chord_spans']:
        if s >= 0:
            if s < prev:
                monotonic = False
                break
            prev = s
    checks['monotonic_spans'] = monotonic
    
    all_ok = all(checks.values())
    
    if verbose:
        status = "PASS" if all_ok else "FAIL"
        print(f"  [{status}] {result['file']}: "
              f"{result['n_tokens']} tokens, {result['n_chords']} chords"
              f"{'' if all_ok else ' — ' + str({k: v for k, v in checks.items() if not v})}")
    
    return checks


# =============================================================================
# CLI / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description="Dual-channel pre-processing for 53-TET MPE MIDI"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Single file test
    sub = subparsers.add_parser("test", help="Test pipeline on a single MIDI file")
    sub.add_argument("midi_file", help="Path to MIDI file")
    
    # Batch preprocess
    sub = subparsers.add_parser("batch", help="Batch pre-process a directory")
    sub.add_argument("midi_dir", help="Directory with MIDI files")
    sub.add_argument("output_dir", help="Output directory")
    sub.add_argument("--max-files", type=int, help="Limit number of files")
    
    # Validate
    sub = subparsers.add_parser("validate", help="Validate pre-processed data")
    sub.add_argument("data_dir", help="Directory with pre-processed .json files")
    
    args = parser.parse_args()
    
    if args.command == "test":
        print("=" * 70)
        print("Dual-Channel Pre-processing — Single File Test")
        print("=" * 70)
        
        tokenizer = MPETokenizer()
        result = preprocess_song(args.midi_file, tokenizer=tokenizer)
        
        if result:
            print(f"\nFile: {result['file']}")
            print(f"  Tokens:     {result['n_tokens']}")
            print(f"  Chords:     {result['n_chords']}")
            print(f"  Avg tok/ch: {result['n_tokens'] / max(1, result['n_chords']):.1f}")
            
            # Show first few chords
            print(f"\n  First 5 chords — EigenSpace 4D:")
            for i, vec in enumerate(result['eigenspace_4d'][:5]):
                a, b, g, d = vec
                print(f"    Chord {i}: α={a:.5f}  β={b:.5f}  γ={g:.5f}  δ={d:.5f}")
            
            # Show token/span alignment for first chord
            print(f"\n  Token alignment (first 2 chords):")
            shown_chords = 0
            for i, (tok, span) in enumerate(
                zip(result['token_strs'][:40], result['chord_spans'][:40])
            ):
                if tok == CHORD_START_TOKEN:
                    shown_chords += 1
                    if shown_chords > 2:
                        break
                chord_str = f"chord[{span}]" if span >= 0 else "──────"
                print(f"    {i:3d}  {tok:16s}  {chord_str}")
            
            # Validate
            print(f"\n  Validation:")
            validate_song(result)
            
            # Test on-the-fly computation
            print(f"\n  On-the-fly EigenSpace (from tokens):")
            eigen_online, spans_online = tokens_to_eigenspace_online(
                result['token_strs']
            )
            print(f"    Chords detected: {len(eigen_online)}")
            match = 0
            for a, b in zip(result['eigenspace_4d'], eigen_online):
                if all(abs(x - y) < 1e-4 for x, y in zip(a, b)):
                    match += 1
            print(f"    Match pre-computed: {match}/{len(eigen_online)} "
                  f"({'100%' if match == len(eigen_online) else 'MISMATCH!'})")
        else:
            print("Failed to process file.")
    
    elif args.command == "batch":
        preprocess_dataset(
            args.midi_dir, args.output_dir,
            max_files=args.max_files,
        )
    
    elif args.command == "validate":
        data_dir = Path(args.data_dir)
        files = sorted(data_dir.glob("*.json"))
        print(f"Validating {len(files)} pre-processed files...")
        
        all_ok = 0
        for f in files:
            with open(f, 'r') as fp:
                song = json.load(fp)
            checks = validate_song(song, verbose=False)
            if all(checks.values()):
                all_ok += 1
            else:
                print(f"  FAIL: {f.name} — {checks}")
        
        print(f"\n{all_ok}/{len(files)} passed all checks.")
    
    else:
        parser.print_help()
