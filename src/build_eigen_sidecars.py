"""
build_eigen_sidecars.py
=======================
Parallel CLI to compute ``.eigen.npy`` sidecars for every paired
(``.mid``, ``.txt``) song under ``dataset/midi_files/53_tet_mpe/``.

One EigenSpace row per chord, shape ``(N_chords, 4)``:
    (α, β, γ, D) — root-invariant, octave-folded.

The output lives next to each MIDI file as ``<stem>.eigen.npy``.

Usage
-----
    python build_eigen_sidecars.py [--workers N] [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MIDI_ROOT = ROOT / "dataset" / "midi_files" / "53_tet_mpe"
TEXT_ROOT = ROOT / "dataset" / "text_files" / "53_tet_mpe"


# Worker-local handle to avoid reloading the 150³ dissonance cube per song.
_DMAP = None


def _init_worker():
    global _DMAP
    from eigenspace import DissonanceMap
    _DMAP = DissonanceMap()


def _process_one(midi_path: str, force: bool) -> tuple[str, int, str | None]:
    """Returns (midi_path, n_chords, error_msg_or_None)."""
    global _DMAP
    try:
        mp = Path(midi_path)
        out_path = mp.with_suffix(".eigen.npy")
        if out_path.exists() and not force:
            arr = np.load(out_path, mmap_mode="r")
            return midi_path, int(arr.shape[0]), None

        text_path = Path(str(mp).replace("/midi_files/", "/text_files/")
                         ).with_suffix(".txt")
        if not text_path.exists():
            return midi_path, 0, f"missing text sidecar: {text_path}"

        import eigen_sidecar
        arr = eigen_sidecar.build_eigen_sidecar(mp, text_path, _DMAP)
        np.save(out_path, arr)
        return midi_path, int(arr.shape[0]), None
    except Exception as e:  # noqa: BLE001
        return midi_path, 0, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N files (0 = all).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if .eigen.npy already exists.")
    args = ap.parse_args()

    # Ensure src/ on path for worker imports
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    pattern = str(MIDI_ROOT / "*" / "*.mid")
    files = sorted(glob.glob(pattern))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no files found under {MIDI_ROOT}")
        sys.exit(1)

    print(f"scanning {len(files):,} files with {args.workers} workers …")
    t0 = time.time()
    ok = 0
    failed: list[tuple[str, str]] = []
    n_chords_total = 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_process_one, f, args.force): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            path, n_c, err = fut.result()
            if err is None:
                ok += 1
                n_chords_total += n_c
            else:
                failed.append((path, err))
            if i % 5000 == 0 or i == len(files):
                dt = time.time() - t0
                rate = i / dt if dt else 0
                eta = (len(files) - i) / rate if rate else 0
                print(f"  {i:>7,}/{len(files):,}  ok={ok:,}  fail={len(failed):,}"
                      f"  {rate:,.0f} f/s  ETA {eta/60:.1f} min")

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min.")
    print(f"  ok:     {ok:,}")
    print(f"  failed: {len(failed):,}")
    print(f"  total chord rows: {n_chords_total:,}")

    if failed:
        sample = failed[:10]
        print("\nFirst failures:")
        for p, e in sample:
            print(f"  {Path(p).name}: {e}")


if __name__ == "__main__":
    main()
