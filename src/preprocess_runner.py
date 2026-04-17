"""
preprocess_runner.py
====================
Parallel runner for dual-channel dataset formation.

Processes all 53-TET MPE MIDI files into the dual-channel format
(MIDI tokens + EigenSpace 4D) using multiprocessing.

Each worker:
  1. Loads its own MPETokenizer (once, at init)
  2. Processes a batch of MIDI files via preprocess_song()
  3. Writes per-song .json files to the output directory

Usage
-----
  # Full dataset, 20 workers:
  python preprocess_runner.py --workers 20

  # Small test run (100 files, 4 workers):
  python preprocess_runner.py --workers 4 --max-files 100

  # Resume (skip already-processed files):
  python preprocess_runner.py --workers 20 --resume
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so workers can import our modules
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# =============================================================================
# WORKER — runs in a subprocess
# =============================================================================

# Per-worker global state (initialised once per process)
_worker_tokenizer = None
_worker_style_lookup = None


def _worker_init(style_lookup_json):
    """
    Initialise heavy resources once per worker process.
    Called automatically by ProcessPoolExecutor via initializer=.
    """
    global _worker_tokenizer, _worker_style_lookup

    # These imports happen inside the worker process
    import importlib
    tok_mod = importlib.import_module("tokenizer")

    _worker_tokenizer = tok_mod.MPETokenizer()
    _worker_style_lookup = json.loads(style_lookup_json) if style_lookup_json else {}


def _worker_process_file(args):
    """
    Process a single MIDI file → write .json.

    Args:
        args: (midi_path_str, output_dir_str)

    Returns:
        (success: bool, file_name: str, n_tokens: int, n_chords: int, error: str|None)
    """
    midi_path_str, output_dir_str = args

    # Late import so the module is resolved inside the worker
    import importlib
    import re as _re
    preproc = importlib.import_module("preprocess")
    tok_mod = importlib.import_module("tokenizer")

    # Extract song name from MIDI filename for style lookup
    # Pattern: NNNNN_SongName_Key_mode_type_X_label.mid
    style_label = None
    if _worker_style_lookup:
        stem = Path(midi_path_str).stem
        m = _re.match(r'\d+_(.+?)_[A-G][b#]?_(major|minor)_type_', stem)
        if m:
            song_name = m.group(1)
            key = _re.sub(r'[^a-z0-9]', '', song_name.lower())
            raw_style = _worker_style_lookup.get(key)
            if raw_style:
                style_label = tok_mod.classify_style(raw_style)

    try:
        result = preproc.preprocess_song(
            midi_path_str,
            tokenizer=_worker_tokenizer,
            style_label=style_label,
        )

        if result is None:
            return (False, Path(midi_path_str).name, 0, 0, "no chords parsed")

        # Write compact JSON (no token_strs — saves ~40% disk)
        compact = {
            "file":          result["file"],
            "type_label":    result.get("type_label"),
            "style_label":   result.get("style_label"),
            "token_ids":     result["token_ids"],
            "eigenspace_4d": result["eigenspace_4d"],
            "chord_spans":   result["chord_spans"],
            "n_chords":      result["n_chords"],
            "n_tokens":      result["n_tokens"],
        }

        song_name = Path(result["file"]).stem
        out_path = Path(output_dir_str) / f"{song_name}.json"
        with open(out_path, "w") as f:
            json.dump(compact, f, separators=(",", ":"))  # compact JSON

        return (True, result["file"], result["n_tokens"], result["n_chords"], None)

    except Exception as e:
        return (False, Path(midi_path_str).name, 0, 0, str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel dual-channel dataset formation for 53-TET MPE MIDI"
    )
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Number of parallel worker processes (default: 20)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of files to process (default: all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files that already have a .json in the output directory"
    )
    parser.add_argument(
        "--midi-dir", type=str,
        default=str(_ROOT_DIR / "dataset" / "midi_files" / "53_tet_mpe"),
        help="Input directory with 53-TET MPE MIDI files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_ROOT_DIR / "dataset" / "preprocessed"),
        help="Output directory for dual-channel data"
    )
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    output_dir = Path(args.output_dir)
    songs_dir = output_dir / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)

    # ── Build style lookup from metadata ──
    import re as _re
    metadata_dir = _ROOT_DIR / "dataset" / "metadata"
    style_lookup = {}  # {normalized_song_name: raw_style}
    if metadata_dir.is_dir():
        for mf in metadata_dir.glob("*.json"):
            try:
                with open(mf, "r") as f:
                    meta = json.load(f)
                raw_style = meta.get("style")
                song_name = meta.get("song_name") or mf.stem
                if raw_style:
                    key = _re.sub(r'[^a-z0-9]', '', song_name.lower())
                    style_lookup[key] = raw_style
            except Exception:
                pass
    style_lookup_json = json.dumps(style_lookup) if style_lookup else ""

    # ── Collect input files ──
    print("=" * 72)
    print("DUAL-CHANNEL DATASET FORMATION")
    print("=" * 72)
    print(f"  Input:   {midi_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Workers: {args.workers}")
    print(f"  Style lookup: {len(style_lookup):,} songs with metadata")

    # rglob to find files in type subdirectories; exclude any 12_tet files in case
    # the parent midi_files/ dir was passed instead of 53_tet_mpe/
    all_files = sorted(
        f for f in midi_dir.rglob("*.mid")
        if "12_tet" not in f.parts
    )
    print(f"  MIDI files found: {len(all_files):,}")

    # ── Resume: skip already-processed ──
    if args.resume:
        existing = set(f.stem for f in songs_dir.glob("*.json"))
        before = len(all_files)
        all_files = [f for f in all_files if f.stem not in existing]
        skipped = before - len(all_files)
        print(f"  Resume: skipping {skipped:,} already processed, {len(all_files):,} remaining")

    # ── Limit ──
    if args.max_files and args.max_files < len(all_files):
        all_files = all_files[:args.max_files]
        print(f"  Limited to: {len(all_files):,} files")

    if not all_files:
        print("\nNo files to process.")
        return

    # ── Prepare tasks ──
    tasks = [
        (str(f), str(songs_dir))
        for f in all_files
    ]

    # ── Process ──
    print(f"\nProcessing {len(tasks):,} files with {args.workers} workers...")
    print("-" * 72)

    t_start = time.time()
    success_count = 0
    fail_count = 0
    total_tokens = 0
    total_chords = 0
    failures = []

    # Use chunksize for efficiency with large task lists
    # ~100 files per dispatch reduces IPC overhead
    chunksize = max(1, min(500, len(tasks) // (args.workers * 4)))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(style_lookup_json,),
    ) as executor:
        # Submit all at once — executor handles queuing
        results_iter = executor.map(
            _worker_process_file, tasks, chunksize=chunksize
        )

        # Progress tracking
        report_every = max(1, len(tasks) // 20)  # ~5% intervals
        processed = 0

        for ok, fname, n_tok, n_ch, err in results_iter:
            processed += 1

            if ok:
                success_count += 1
                total_tokens += n_tok
                total_chords += n_ch
            else:
                fail_count += 1
                if len(failures) < 20:
                    failures.append(f"{fname}: {err}")

            # Progress report
            if processed % report_every == 0 or processed == len(tasks):
                elapsed = time.time() - t_start
                rate = processed / elapsed
                eta = (len(tasks) - processed) / rate if rate > 0 else 0
                pct = processed / len(tasks) * 100
                print(
                    f"  [{pct:5.1f}%] {processed:>7,}/{len(tasks):,}  "
                    f"ok={success_count:,}  fail={fail_count:,}  "
                    f"{rate:,.0f} files/s  ETA {eta:.0f}s"
                )

    elapsed = time.time() - t_start

    # ── Summary ──
    print("-" * 72)
    print(f"\n  Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {success_count:,} / {len(tasks):,}")
    print(f"  Failed:  {fail_count:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total chords: {total_chords:,}")
    if success_count > 0:
        print(f"  Avg tokens/song: {total_tokens / success_count:.1f}")
        print(f"  Avg chords/song: {total_chords / success_count:.1f}")
        print(f"  Throughput: {success_count / elapsed:,.0f} files/s")

    if failures:
        print(f"\n  Sample failures ({min(len(failures), 20)} of {fail_count}):")
        for f in failures[:20]:
            print(f"    - {f}")

    # ── Write index + stats ──
    print(f"\nWriting index and stats...")

    # Index: list all processed songs
    song_jsons = sorted(songs_dir.glob("*.json"))
    index = []
    for sj in song_jsons:
        try:
            with open(sj, "r") as f:
                header = json.load(f)
            index.append({
                "file": header["file"],
                "n_tokens": header["n_tokens"],
                "n_chords": header["n_chords"],
            })
        except Exception:
            pass  # skip corrupt files

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Stats
    stats = {
        "total_files_in_dir": len(all_files),
        "processed": success_count,
        "failed": fail_count,
        "total_tokens": total_tokens,
        "total_chords": total_chords,
        "elapsed_seconds": round(elapsed, 1),
        "workers": args.workers,
        "songs_in_index": len(index),
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save vocab
    import importlib
    tok_mod = importlib.import_module("tokenizer")
    tokenizer = tok_mod.MPETokenizer()
    tokenizer.save_vocab(str(output_dir / "vocab.json"))

    print(f"  index.json      ({len(index):,} songs)")
    print(f"  dataset_stats.json")
    print(f"  vocab.json      ({tokenizer.vocab_size} tokens)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
