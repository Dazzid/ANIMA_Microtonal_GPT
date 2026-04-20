"""
pack_data_v3.py
===============
Phase-A v3 data packer: paired ``(.txt, .mid, .eigen.npy)`` → train/val ``.bin``.

This is the single-stage replacement for the old
``preprocess.py`` → ``pack_data.py`` pipeline.

Pipeline
--------
    for each paired song:
        tokens, spans = l1.build_merged_tokens_with_spans(.mid, .txt, tokenizer)
        ids           = tokenizer.encode_to_ids(tokens)
        eigen_chord   = np.load("<stem>.eigen.npy")            # (N_chords, 4)
        eigen_token   = expand(eigen_chord, spans)             # (T, 4)
    chunk to block_size + 1, split songs train/val, write .bin.

Output (``dataset/tokenized/`` by default)
------------------------------------------
    train_tokens.bin   uint16   (N_train_seqs, block_size+1)
    train_eigen.bin    float16  (N_train_seqs, block_size+1, 4)
    val_tokens.bin     uint16   (N_val_seqs,   block_size+1)
    val_eigen.bin      float16  (N_val_seqs,   block_size+1, 4)
    meta.json
    vocab.json

Songs lacking a ``.eigen.npy`` sidecar (the 0.05 % that failed the L1/L2
merge) are skipped with a count report.

Usage
-----
    python pack_data_v3.py                         # full dataset
    python pack_data_v3.py --max-files 100         # quick sanity
    python pack_data_v3.py --block-size 2048       # shorter context
    python pack_data_v3.py --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# =============================================================================
# CONSTANTS — must agree with model.py and tokenizer.py
# =============================================================================

PAD_ID = 0
HEADER_EIGEN = np.array([1.0, 1.0, 2.0, 0.0], dtype=np.float16)  # non-chord default


# =============================================================================
# PER-SONG WORKER
# =============================================================================


def _process_pair(args) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(token_ids uint16, eigen_per_token float16)`` or ``None`` on failure."""
    midi_path, text_path, eigen_path = args
    try:
        import l1
        import tokenizer as _tokmod
        tok = _process_pair._tok  # pre-built tokenizer cached in worker
        tokens, spans = l1.build_merged_tokens_with_spans(midi_path, text_path, tok)
        ids = tok.encode_to_ids(tokens)
        ids = np.asarray(ids, dtype=np.uint16)

        eigen_chord = np.load(eigen_path).astype(np.float16)  # (N_chords, 4)
        n_tokens = len(tokens)
        eigen_tok = np.tile(HEADER_EIGEN, (n_tokens, 1))      # (T, 4) fp16

        for i, k in enumerate(spans):
            if 0 <= k < len(eigen_chord):
                eigen_tok[i] = eigen_chord[k]
            # else k == -1 → keep HEADER_EIGEN
        return ids, eigen_tok
    except Exception:
        return None


def _worker_init():
    import tokenizer as _tokmod
    _process_pair._tok = _tokmod.MPETokenizer()


# =============================================================================
# CHUNKING
# =============================================================================


def _chunk_song(tokens: np.ndarray, eigen: np.ndarray, seq_len: int
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a song into fixed-length sequences. Last chunk padded if short."""
    n = len(tokens)
    chunks = []
    for start in range(0, n, seq_len):
        end = start + seq_len
        tok_chunk = np.full(seq_len, PAD_ID, dtype=np.uint16)
        eig_chunk = np.tile(HEADER_EIGEN, (seq_len, 1))
        actual = min(end, n)
        L = actual - start
        tok_chunk[:L] = tokens[start:actual]
        eig_chunk[:L] = eigen[start:actual]
        chunks.append((tok_chunk, eig_chunk))
    return chunks


# =============================================================================
# BINARY WRITER
# =============================================================================


def _write_binary(arr: np.ndarray, path: Path) -> None:
    size_mb = arr.nbytes / (1024 * 1024)
    print(f"  writing {path.name}: {arr.shape} {arr.dtype}  ({size_mb:,.1f} MB)",
          flush=True)
    with open(path, "wb") as f:
        arr.tofile(f)


# =============================================================================
# STREAMING EIGEN STATS (low-memory)
# =============================================================================


def _eigen_stats_streaming(path: Path, seq_len: int, n_dims: int = 4,
                           chunk_seqs: int = 8192) -> dict:
    """Compute per-dim mean/std of a float16 eigen .bin without loading it all.

    Two-pass (mean, then variance) for numerical stability. Accumulates in
    float64. Memory footprint ≈ ``chunk_seqs * seq_len * n_dims * 4`` bytes
    (≈1 GB at defaults).
    """
    nbytes = path.stat().st_size
    itemsize = np.dtype(np.float16).itemsize
    n_elems = nbytes // itemsize
    if n_elems % (seq_len * n_dims) != 0:
        raise ValueError(
            f"{path.name}: {n_elems} fp16 elems not divisible by "
            f"seq_len*n_dims = {seq_len*n_dims}"
        )
    n_seqs = n_elems // (seq_len * n_dims)
    mm = np.memmap(path, dtype=np.float16, mode="r",
                   shape=(n_seqs, seq_len, n_dims))

    total = n_seqs * seq_len  # samples per dim
    sums = np.zeros(n_dims, dtype=np.float64)
    for s in range(0, n_seqs, chunk_seqs):
        blk = mm[s:s + chunk_seqs].astype(np.float32)
        sums += blk.reshape(-1, n_dims).sum(axis=0, dtype=np.float64)
    means = sums / total

    sqd = np.zeros(n_dims, dtype=np.float64)
    for s in range(0, n_seqs, chunk_seqs):
        blk = mm[s:s + chunk_seqs].astype(np.float32)
        diff = blk.reshape(-1, n_dims) - means  # broadcast
        sqd += (diff * diff).sum(axis=0, dtype=np.float64)
    stds = np.sqrt(sqd / total)

    del mm
    return {
        "n_seqs": int(n_seqs),
        "n_samples": int(total),
        "means": means.tolist(),
        "stds":  stds.tolist(),
    }


def _write_meta(output_dir: Path, args, block_size: int, n_songs_total: int,
                n_train_songs: int, n_val_songs: int,
                n_train_seqs: int, n_val_seqs: int,
                missing_eigen: int, missing_text: int, failed: int,
                vocab_size: int) -> None:
    """Write ``meta.json`` using streaming stats over the train eigen bin."""
    seq_len = block_size + 1
    print("\ncomputing eigenspace stats (streaming, train split only)…",
          flush=True)
    t1 = time.time()
    stats = _eigen_stats_streaming(output_dir / "train_eigen.bin", seq_len)
    print(f"  stats done in {time.time()-t1:.1f}s  "
          f"({stats['n_seqs']:,} seqs, {stats['n_samples']:,} samples/dim)",
          flush=True)

    dim_names = ["alpha", "beta", "gamma", "D"]
    eigen_stats = {
        name: {"mean": stats["means"][i], "std": stats["stds"][i]}
        for i, name in enumerate(dim_names)
    }

    n_train_tokens = n_train_seqs * seq_len
    n_val_tokens = n_val_seqs * seq_len
    meta = {
        "vocab_size":        vocab_size,
        "block_size":        block_size,
        "seq_len":           seq_len,
        "val_ratio":         args.val_ratio,
        "seed":              args.seed,
        "n_songs_total":     n_songs_total,
        "n_train_songs":     n_train_songs,
        "n_val_songs":       n_val_songs,
        "n_train_seqs":      int(n_train_seqs),
        "n_val_seqs":        int(n_val_seqs),
        "n_train_tokens":    int(n_train_tokens),
        "n_val_tokens":      int(n_val_tokens),
        "eigenspace_dims":   4,
        "eigenspace_order":  dim_names,
        "eigenspace_stats":  eigen_stats,
        "dtype_tokens":      "uint16",
        "dtype_eigen":       "float16",
        "skipped_missing_eigen": missing_eigen,
        "skipped_missing_text":  missing_text,
        "skipped_merge_failure": failed,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {output_dir/'meta.json'}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi-dir", default=str(_ROOT_DIR / "dataset" / "midi_files" / "53_tet_mpe"))
    ap.add_argument("--text-dir", default=str(_ROOT_DIR / "dataset" / "text_files" / "53_tet_mpe"))
    ap.add_argument("--output-dir", default=str(_ROOT_DIR / "dataset" / "tokenized"))
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--regen-meta-only", action="store_true",
                    help="Skip packing; recompute meta.json from existing .bin files.")
    args = ap.parse_args()

    midi_dir = Path(args.midi_dir)
    text_dir = Path(args.text_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PACK v3 — paired (.txt, .mid, .eigen.npy) → .bin")
    print("=" * 72)
    print(f"  MIDI: {midi_dir}")
    print(f"  TEXT: {text_dir}")
    print(f"  OUT:  {output_dir}")
    print(f"  block_size: {args.block_size}  val_ratio: {args.val_ratio}  seed: {args.seed}")

    # ── Regen meta.json only (bins already on disk) ──
    if args.regen_meta_only:
        import tokenizer as _tokmod
        tok = _tokmod.MPETokenizer()
        tok.save_vocab(str(output_dir / "vocab.json"))

        seq_len = args.block_size + 1
        fp16 = np.dtype(np.float16).itemsize
        u16 = np.dtype(np.uint16).itemsize
        train_eigen_path = output_dir / "train_eigen.bin"
        val_eigen_path = output_dir / "val_eigen.bin"
        if not train_eigen_path.exists() or not val_eigen_path.exists():
            raise SystemExit(
                f"--regen-meta-only: missing bin files in {output_dir}"
            )
        n_train_seqs = train_eigen_path.stat().st_size // (fp16 * seq_len * 4)
        n_val_seqs = val_eigen_path.stat().st_size // (fp16 * seq_len * 4)
        tr_tok_path = output_dir / "train_tokens.bin"
        va_tok_path = output_dir / "val_tokens.bin"
        n_train_seqs_t = tr_tok_path.stat().st_size // (u16 * seq_len)
        n_val_seqs_t = va_tok_path.stat().st_size // (u16 * seq_len)
        if n_train_seqs != n_train_seqs_t or n_val_seqs != n_val_seqs_t:
            raise SystemExit(
                f"bin shape mismatch: eigen train={n_train_seqs} tok train={n_train_seqs_t}, "
                f"eigen val={n_val_seqs} tok val={n_val_seqs_t}"
            )
        print(f"  train seqs: {n_train_seqs:,}  val seqs: {n_val_seqs:,}")
        _write_meta(
            output_dir, args, block_size=args.block_size,
            n_songs_total=n_train_seqs + n_val_seqs,  # unknown; use seq count as proxy
            n_train_songs=-1, n_val_songs=-1,
            n_train_seqs=n_train_seqs, n_val_seqs=n_val_seqs,
            missing_eigen=-1, missing_text=-1, failed=-1,
            vocab_size=tok.vocab_size,
        )
        return

    # ── 1. Collect paired songs that have an .eigen.npy sidecar ──
    midi_files = sorted(glob.glob(str(midi_dir / "*" / "*.mid")))
    if args.max_files:
        midi_files = midi_files[: args.max_files]

    triples: list[tuple[str, str, str]] = []
    missing_eigen = 0
    missing_text = 0
    for m in midi_files:
        eigen = m[:-4] + ".eigen.npy"
        txt = m.replace(str(midi_dir), str(text_dir)).replace(".mid", ".txt")
        if not os.path.exists(eigen):
            missing_eigen += 1
            continue
        if not os.path.exists(txt):
            missing_text += 1
            continue
        triples.append((m, txt, eigen))

    print(f"  found {len(midi_files):,} MIDI files")
    print(f"  skipped (no .eigen.npy): {missing_eigen:,}")
    print(f"  skipped (no .txt):       {missing_text:,}")
    print(f"  paired songs to pack:    {len(triples):,}")
    if not triples:
        print("nothing to pack.")
        return

    # ── 2. Parallel load + expand ──
    t0 = time.time()
    all_tokens: list[np.ndarray] = []
    all_eigen: list[np.ndarray] = []
    failed = 0
    CHUNK = max(1, min(500, len(triples) // (args.workers * 4)))
    print(f"\nloading + expanding with {args.workers} workers (chunksize={CHUNK})…")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as ex:
        for i, res in enumerate(ex.map(_process_pair, triples, chunksize=CHUNK), 1):
            if res is None:
                failed += 1
            else:
                ids, eigen_tok = res
                all_tokens.append(ids)
                all_eigen.append(eigen_tok)
            if i % 20000 == 0 or i == len(triples):
                dt = time.time() - t0
                rate = i / dt if dt else 0
                print(f"  {i:>7,}/{len(triples):,}  ok={len(all_tokens):,}  "
                      f"fail={failed:,}  {rate:,.0f} f/s")

    n = len(all_tokens)
    print(f"\nloaded {n:,} songs  (fail={failed:,})  in {(time.time()-t0)/60:.1f} min")

    # ── 3. Train/val split (by song) ──
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_set = set(perm[:n_val].tolist())
    n_train = n - n_val
    print(f"\nsplit: {n_train:,} train / {n_val:,} val songs")

    # ── 4. Chunk + stack ──
    seq_len = args.block_size + 1
    tr_tok, tr_eig, va_tok, va_eig = [], [], [], []
    for i in range(n):
        for tc, ec in _chunk_song(all_tokens[i], all_eigen[i], seq_len):
            (va_tok if i in val_set else tr_tok).append(tc)
            (va_eig if i in val_set else tr_eig).append(ec)

    train_tokens = np.stack(tr_tok)          # (N_tr, seq_len) uint16
    train_eigen = np.stack(tr_eig)           # (N_tr, seq_len, 4) float16
    val_tokens = np.stack(va_tok)
    val_eigen = np.stack(va_eig)
    print(f"  train seqs: {len(train_tokens):,}  val seqs: {len(val_tokens):,}")

    # ── 5. Write binaries ──
    print()
    _write_binary(train_tokens, output_dir / "train_tokens.bin")
    _write_binary(train_eigen,  output_dir / "train_eigen.bin")
    _write_binary(val_tokens,   output_dir / "val_tokens.bin")
    _write_binary(val_eigen,    output_dir / "val_eigen.bin")

    n_train_seqs = int(len(train_tokens))
    n_val_seqs = int(len(val_tokens))

    # Free the huge in-memory arrays before streaming stats — the training
    # eigen tensor alone would be ~43 GB as float32.
    del train_tokens, train_eigen, val_tokens, val_eigen, tr_tok, tr_eig, va_tok, va_eig
    import gc
    gc.collect()

    # ── 6. Vocab + meta ──
    import tokenizer as _tokmod
    tok = _tokmod.MPETokenizer()
    tok.save_vocab(str(output_dir / "vocab.json"))

    _write_meta(
        output_dir, args, block_size=args.block_size,
        n_songs_total=n, n_train_songs=n_train, n_val_songs=n_val,
        n_train_seqs=n_train_seqs, n_val_seqs=n_val_seqs,
        missing_eigen=missing_eigen, missing_text=missing_text, failed=failed,
        vocab_size=tok.vocab_size,
    )

    # ── 7. Summary ──
    total_mb = sum((output_dir / f).stat().st_size for f in [
        "train_tokens.bin", "train_eigen.bin", "val_tokens.bin", "val_eigen.bin"
    ]) / (1024 * 1024)
    with open(output_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"\n{'='*72}")
    print(f"DONE — {total_mb:,.1f} MB on disk, {time.time()-t0:.0f}s total")
    print(f"  eigenspace (train) stats:")
    for k, v in meta["eigenspace_stats"].items():
        print(f"    {k}: {v['mean']:8.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
