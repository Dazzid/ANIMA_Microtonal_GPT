"""
pack_data.py
============
Converts preprocessed dual-channel JSONs into memory-mapped binary files
optimized for GPT-2 training.

Dual-channel output
-------------------
  Channel 1 — token_ids   (uint16)   → token embedding → attention mechanism
  Channel 2 — eigenspace  (float32×4) → spatial embedding  (harmonic geometry)

Pipeline
--------
  1. Load all per-song .json files from dataset/preprocessed/songs/
  2. Expand eigenspace_4d from chord-level → token-level via chord_spans
  3. Concatenate ALL songs into one long stream (separated by <end> tokens)
  4. Split into train / validation sets (by songs, not by tokens)
  5. Write memory-mapped .bin files for zero-copy DataLoader access

Output files
------------
  dataset/tokenized/
    train_tokens.bin    — uint16,   shape (N_train,)
    train_eigen.bin     — float32,  shape (N_train, 4)
    val_tokens.bin      — uint16,   shape (N_val,)
    val_eigen.bin       — float32,  shape (N_val, 4)
    meta.json           — vocab_size, block_size, split sizes, stats
    vocab.json          — copy of the vocabulary (for inference)

Usage
-----
  # Full dataset:
  python pack_data.py

  # Quick test (100 songs):
  python pack_data.py --max-files 100

  # Custom split ratio:
  python pack_data.py --val-ratio 0.05

  # Custom block size:
  python pack_data.py --block-size 1024
"""

import argparse
import json
import numpy as np
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

# ── Path setup ──
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# =============================================================================
# CONSTANTS
# =============================================================================

# Default eigenspace vector for non-chord tokens (e.g. <start>, BAR, <end>)
DEFAULT_EIGEN = np.array([1.0, 1.0, 2.0, 1.0], dtype=np.float32)

# Token IDs for special markers (from the vocabulary)
PAD_ID = 0
START_ID = 1
END_ID = 2


# =============================================================================
# SONG EXPANSION — expand eigenspace from chord-level to token-level
# =============================================================================

def expand_song(song: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand a single song JSON into aligned token + eigenspace arrays.

    Takes the per-chord eigenspace_4d and broadcasts it to every token
    position using the chord_spans mapping.

    Args:
        song: Dict with keys 'token_ids', 'eigenspace_4d', 'chord_spans'

    Returns:
        token_ids:  np.ndarray, shape (n_tokens,), dtype uint16
        eigenspace: np.ndarray, shape (n_tokens, 4), dtype float16
    """
    token_ids = np.array(song['token_ids'], dtype=np.uint16)
    n_tokens = len(token_ids)

    chord_spans = song['chord_spans']
    # Prefer 4D data; fall back to legacy 3D
    eigen_key = 'eigenspace_4d' if 'eigenspace_4d' in song else 'eigenspace_3d'
    eigen_chords = np.array(song[eigen_key], dtype=np.float16)
    # Pad legacy 3D data with default δ=1.0
    if eigen_chords.ndim == 2 and eigen_chords.shape[1] == 3:
        delta_col = np.full((eigen_chords.shape[0], 1), 1.0, dtype=np.float16)
        eigen_chords = np.concatenate([eigen_chords, delta_col], axis=1)

    # Expand eigenspace to per-token
    eigenspace = np.tile(DEFAULT_EIGEN.astype(np.float16), (n_tokens, 1))  # (n_tokens, 4)

    for i, chord_idx in enumerate(chord_spans):
        if 0 <= chord_idx < len(eigen_chords):
            eigenspace[i] = eigen_chords[chord_idx]

    return token_ids, eigenspace


def load_and_expand_song(json_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load a song JSON file and return expanded (token_ids, eigenspace).
    Returns None on error.
    """
    try:
        with open(json_path, 'r') as f:
            song = json.load(f)
        return expand_song(song)
    except Exception as e:
        return None


# =============================================================================
# BATCH LOADER — parallel JSON loading
# =============================================================================

def load_songs_parallel(
    song_files: List[Path],
    workers: int = 8,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load and expand all songs in parallel.

    Returns:
        all_tokens: list of uint16 arrays
        all_eigen:  list of float32 arrays (each shape (n, 4))
    """
    all_tokens = []
    all_eigen = []
    failed = 0
    total = len(song_files)

    if verbose:
        print(f"  Loading {total:,} songs with {workers} workers...")

    t0 = time.time()

    # Use ProcessPoolExecutor for true parallelism on JSON parsing
    paths_str = [str(f) for f in song_files]
    chunksize = max(1, min(500, total // (workers * 4)))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(load_and_expand_song, paths_str, chunksize=chunksize))

    for result in results:
        if result is not None:
            tokens, eigen = result
            all_tokens.append(tokens)
            all_eigen.append(eigen)
        else:
            failed += 1

    elapsed = time.time() - t0
    if verbose:
        n = len(all_tokens)
        total_tok = sum(len(t) for t in all_tokens)
        print(f"  Loaded {n:,} songs ({failed:,} failed) in {elapsed:.1f}s")
        print(f"  Total tokens: {total_tok:,}")

    return all_tokens, all_eigen


# =============================================================================
# CONCATENATION + SPLIT
# =============================================================================

def _chunk_song(tokens: np.ndarray, eigen: np.ndarray,
                seq_len: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a single song into fixed-length sequences of exactly `seq_len` tokens.

    Songs shorter than seq_len are padded with <pad> (id=0) and DEFAULT_EIGEN.
    Songs longer than seq_len are split into multiple chunks; the last chunk
    is padded if needed.

    Returns:
        List of (token_chunk, eigen_chunk) tuples, each of shape
        (seq_len,) uint16 and (seq_len, 4) float16.
    """
    n = len(tokens)
    chunks = []
    for start in range(0, n, seq_len):
        end = start + seq_len
        tok_chunk = np.full(seq_len, PAD_ID, dtype=np.uint16)
        eig_chunk = np.tile(DEFAULT_EIGEN.astype(np.float16), (seq_len, 1))

        actual_end = min(end, n)
        length = actual_end - start
        tok_chunk[:length] = tokens[start:actual_end]
        eig_chunk[:length] = eigen[start:actual_end]

        chunks.append((tok_chunk, eig_chunk))
    return chunks


def concatenate_and_split(
    all_tokens: List[np.ndarray],
    all_eigen: List[np.ndarray],
    val_ratio: float = 0.02,
    block_size: int = 4096,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Shuffle songs, split into train/val, chunk into fixed-length padded
    sequences, and stack into 2D arrays.

    Each song becomes one or more sequences of exactly (block_size + 1) tokens.
    The +1 allows the (input, target) shift: x = seq[:block_size], y = seq[1:block_size+1].
    Songs shorter than (block_size + 1) are padded with <pad> (id=0).
    Songs longer are split into multiple chunks.

    The split is done at the SONG level (not token level) to ensure
    the validation set contains entirely unseen songs.

    Returns:
        Dict with keys:
          'train_tokens', 'train_eigen',   — shape (N_train_seqs, block_size+1)
          'val_tokens', 'val_eigen',       — shape (N_val_seqs, block_size+1)
          'n_train_songs', 'n_val_songs',
          'n_train_seqs', 'n_val_seqs'
    """
    seq_len = block_size + 1  # +1 for the target shift

    n = len(all_tokens)
    rng = np.random.RandomState(seed)

    # Shuffle song indices
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    val_idx = set(indices[:n_val].tolist())

    if verbose:
        print(f"\n  Split: {n_train:,} train / {n_val:,} val songs "
              f"(ratio={val_ratio:.3f}, seed={seed})")
        print(f"  Sequence length: {seq_len} (block_size={block_size} + 1)")

    # Chunk each song and assign to train/val
    train_tok_chunks = []
    train_eig_chunks = []
    val_tok_chunks = []
    val_eig_chunks = []

    for i in range(n):
        chunks = _chunk_song(all_tokens[i], all_eigen[i], seq_len)
        if i in val_idx:
            for tc, ec in chunks:
                val_tok_chunks.append(tc)
                val_eig_chunks.append(ec)
        else:
            for tc, ec in chunks:
                train_tok_chunks.append(tc)
                train_eig_chunks.append(ec)

    # Stack into 2D arrays
    train_tokens = np.stack(train_tok_chunks)   # (N_train_seqs, seq_len)
    train_eigen = np.stack(train_eig_chunks)    # (N_train_seqs, seq_len, 4)
    val_tokens = np.stack(val_tok_chunks)
    val_eigen = np.stack(val_eig_chunks)

    if verbose:
        print(f"  Train: {len(train_tokens):,} sequences ({train_tokens.size:,} tokens incl. padding)")
        print(f"  Val:   {len(val_tokens):,} sequences ({val_tokens.size:,} tokens incl. padding)")

    return {
        'train_tokens': train_tokens,
        'train_eigen': train_eigen,
        'val_tokens': val_tokens,
        'val_eigen': val_eigen,
        'n_train_songs': n_train,
        'n_val_songs': n_val,
        'n_train_seqs': len(train_tokens),
        'n_val_seqs': len(val_tokens),
    }


# =============================================================================
# BINARY WRITER — memory-mapped format
# =============================================================================

def write_binary(arr: np.ndarray, path: Path, verbose: bool = True, chunk_mb: int = 256):
    """Write a numpy array to a raw binary file in chunks with progress."""
    total_bytes = arr.nbytes
    size_mb = total_bytes / (1024 * 1024)
    chunk_bytes = chunk_mb * 1024 * 1024
    
    if verbose:
        print(f"  Writing {path.name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)...", flush=True)
    
    flat = arr.view(np.uint8).ravel()  # view as raw bytes
    written = 0
    with open(str(path), 'wb') as f:
        while written < len(flat):
            end = min(written + chunk_bytes, len(flat))
            f.write(flat[written:end].tobytes())
            written = end
            if verbose:
                pct = written / len(flat) * 100
                print(f"    {pct:5.1f}%  ({written / (1024**2):.0f} / {size_mb:.0f} MB)", flush=True)
    
    if verbose:
        print(f"  Done: {path.name}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize preprocessed dual-channel data for GPT-2 training"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=str(_ROOT_DIR / "dataset" / "preprocessed"),
        help="Input directory (must contain songs/ and vocab.json)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_ROOT_DIR / "dataset" / "tokenized"),
        help="Output directory for binary training files"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.02,
        help="Fraction of songs for validation (default: 0.02)"
    )
    parser.add_argument(
        "--block-size", type=int, default=4096,
        help="Context window size for training (default: 4096)"
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel workers for loading (default: 16)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of songs (for testing)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    songs_dir = input_dir / "songs"
    vocab_path = input_dir / "vocab.json"

    # ── Validate input ──
    assert songs_dir.exists(), f"Songs directory not found: {songs_dir}"
    assert vocab_path.exists(), f"Vocabulary not found: {vocab_path}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vocab ──
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab['token_to_id'])

    # ── Header ──
    print("=" * 72)
    print("TOKENIZE FOR GPT-2 TRAINING")
    print("=" * 72)
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Block size: {args.block_size}")
    print(f"  Val ratio:  {args.val_ratio}")
    print(f"  Workers:    {args.workers}")

    # ── Collect song files ──
    song_files = sorted(songs_dir.glob("*.json"))
    if args.max_files:
        song_files = song_files[:args.max_files]
    print(f"  Songs:      {len(song_files):,}")

    t_start = time.time()

    # ── Step 1: Load + expand all songs ──
    print(f"\n{'─'*72}")
    print("Step 1/3: Loading and expanding eigenspace...")
    print(f"{'─'*72}")

    all_tokens, all_eigen = load_songs_parallel(
        song_files, workers=args.workers, verbose=True
    )

    # ── Step 2: Split + concatenate ──
    print(f"\n{'─'*72}")
    print("Step 2/3: Train/val split and concatenation...")
    print(f"{'─'*72}")

    data = concatenate_and_split(
        all_tokens, all_eigen,
        val_ratio=args.val_ratio,
        block_size=args.block_size,
        seed=args.seed,
        verbose=True,
    )

    # ── Step 3: Write binary files ──
    print(f"\n{'─'*72}")
    print("Step 3/3: Writing binary files...")
    print(f"{'─'*72}")

    write_binary(data['train_tokens'], output_dir / "train_tokens.bin")
    write_binary(data['train_eigen'],  output_dir / "train_eigen.bin")
    write_binary(data['val_tokens'],   output_dir / "val_tokens.bin")
    write_binary(data['val_eigen'],    output_dir / "val_eigen.bin")

    # Copy vocab
    shutil.copy2(vocab_path, output_dir / "vocab.json")
    print(f"  Copied vocab.json")

    # ── Compute statistics ──
    train_tok = data['train_tokens']
    val_tok = data['val_tokens']

    # Token frequency distribution
    token_counts = np.bincount(train_tok.astype(np.int32), minlength=vocab_size)
    id_to_token = {v: k for k, v in vocab['token_to_id'].items()}

    # Top-20 most common tokens
    top20_ids = np.argsort(token_counts)[::-1][:20]
    top20 = [
        {"id": int(tid), "token": id_to_token.get(int(tid), "?"), "count": int(token_counts[tid])}
        for tid in top20_ids
    ]

    # Eigenspace statistics (compute in float32 for precision)
    te = data['train_eigen'].astype(np.float32)
    eigen_stats = {
        "alpha": {"mean": float(te[:, :, 0].mean()), "std": float(te[:, :, 0].std())},
        "beta":  {"mean": float(te[:, :, 1].mean()), "std": float(te[:, :, 1].std())},
        "gamma": {"mean": float(te[:, :, 2].mean()), "std": float(te[:, :, 2].std())},
        "delta": {"mean": float(te[:, :, 3].mean()), "std": float(te[:, :, 3].std())},
    }

    # ── Write metadata ──
    meta = {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "seq_len": args.block_size + 1,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_train_songs": data['n_train_songs'],
        "n_val_songs": data['n_val_songs'],
        "n_train_seqs": data['n_train_seqs'],
        "n_val_seqs": data['n_val_seqs'],
        "n_train_tokens": int(train_tok.size),
        "n_val_tokens": int(val_tok.size),
        "total_tokens": int(train_tok.size + val_tok.size),
        "eigenspace_dims": 4,
        "eigenspace_stats": eigen_stats,
        "top_20_tokens": top20,
        "dtype_tokens": "uint16",
        "dtype_eigen": "float16",
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Summary ──
    elapsed = time.time() - t_start
    total_mb = sum(
        (output_dir / f).stat().st_size
        for f in ["train_tokens.bin", "train_eigen.bin", "val_tokens.bin", "val_eigen.bin"]
    ) / (1024 * 1024)

    print(f"\n{'='*72}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*72}")
    print(f"  Train: {data['n_train_songs']:,} songs → {data['n_train_seqs']:,} sequences ({train_tok.size:,} tokens)")
    print(f"  Val:   {data['n_val_songs']:,} songs → {data['n_val_seqs']:,} sequences ({val_tok.size:,} tokens)")
    print(f"  Disk:  {total_mb:.1f} MB total")
    print(f"\n  EigenSpace distribution (train):")
    print(f"    α: {eigen_stats['alpha']['mean']:.4f} ± {eigen_stats['alpha']['std']:.4f}")
    print(f"    β: {eigen_stats['beta']['mean']:.4f} ± {eigen_stats['beta']['std']:.4f}")
    print(f"    γ: {eigen_stats['gamma']['mean']:.4f} ± {eigen_stats['gamma']['std']:.4f}")
    print(f"    δ: {eigen_stats['delta']['mean']:.4f} ± {eigen_stats['delta']['std']:.4f}")
    print(f"\n  Files written to {output_dir}/:")
    print(f"    train_tokens.bin  train_eigen.bin")
    print(f"    val_tokens.bin    val_eigen.bin")
    print(f"    meta.json         vocab.json")


if __name__ == "__main__":
    main()
