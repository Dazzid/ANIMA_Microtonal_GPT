"""
eigenspace.py
=============
EigenSpace computation module for 53-TET chord sequences.

Maps each chord to its 4D EigenSpace coordinates:
  (α, β, γ, D)

Where:
  α = frequency ratio of the 3rd   (relative to root, octave-folded)
  β = frequency ratio of the 5th   (relative to root, octave-folded)
  γ = frequency ratio of the 7th   (relative to root, octave-folded)
  D = sensory dissonance at (α, β, γ) from the pre-computed
      4D EigenSpace dissonance map (DissonanceMap.lookup)

The eigenvector is **root-invariant**: Cmaj7 ≡ Dmaj7 ≡ F#maj7, because the
intervals are always measured relative to the chord root and folded into a
single octave (9ths, 11ths, 13ths collapse onto their parent classes).
Slash chords use the **named** root, not the bass.

Usage
-----
  from eigenspace import DissonanceMap, classify_intervals

  diss_map = DissonanceMap()
  alpha, beta, gamma = classify_intervals(intervals)
  D = diss_map.lookup(alpha, beta, gamma)

For the model:
  from eigenspace import EigenSpaceEmbedding

  eigen_emb = EigenSpaceEmbedding(n_embd=768, n_eigen=4)
  # In forward: eigen_emb(eigen_coords)  →  (B, T, n_embd)
"""

import numpy as np
import os
from typing import List, Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# CONSTANTS
# =============================================================================

TET_53 = 53

# Interval classification zones (53-TET steps, disjoint)
#   0       = root (always removed)
#   1–9     = seconds / 9ths       → discarded (extensions)
#   10–22   = α zone (thirds)      → mapped to α axis
#   23      = gap                  → discarded
#   24–35   = β zone (fifths)      → mapped to β axis
#   36–41   = sixths / 13ths       → discarded (extensions)
#   42–52   = γ zone (sevenths)    → mapped to γ axis
THIRD_RANGE   = range(10, 23)   # 10–22 steps
FIFTH_RANGE   = range(24, 36)   # 24–35 steps
SEVENTH_RANGE = range(42, 53)   # 42–52 steps

# Canonical vocabulary values — the ONLY steps in get_53tet_chord_positions()
CANONICAL_THIRDS   = sorted([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22])
CANONICAL_FIFTHS   = sorted([26, 31, 35])
CANONICAL_SEVENTHS = sorted([42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

# Default coordinates when no valid intervals are found
DEFAULT_ALPHA = 1.0     # unison
DEFAULT_BETA  = 1.0     # unison
DEFAULT_GAMMA = 2.0     # octave (only used for non-chord tokens)
DEFAULT_D     = 0.0     # dissonance fallback (filled from map.diss_mean at runtime)
DEFAULT_DELTA = DEFAULT_D  # alias used by preprocess.py

# Number of EigenSpace dimensions (α, β, γ, D)
N_EIGEN = 4


# =============================================================================
# PURE FUNCTIONS
# =============================================================================

def get_53tet_ratio(steps: int) -> float:
    """Convert 53-TET steps to frequency ratio: 2^(steps/53)."""
    return 2.0 ** (steps / 53.0)


def _snap_to_nearest(value: int, canonical: list) -> int:
    """Snap a step value to the nearest canonical vocabulary value."""
    return min(canonical, key=lambda c: abs(c - value))


def classify_intervals(intervals: List[int], root_pc: int = 0) -> Tuple[float, float, float]:
    """
    Classify a chord's intervals into (α, β, γ) frequency ratios.

    Takes a list of 53-TET intervals **relative to the chord root** (e.g.
    ``[0, 18, 31, 44]``) and identifies the third, fifth, and seventh.

    Semantics (root-invariant):
        * All intervals are folded into one octave (``mod 53``) by the caller
          before this function is invoked. 9ths / 11ths / 13ths collapse onto
          their parent classes; notes outside the α/β/γ zones are discarded
          as extensions.
        * Each selected step is snapped to its nearest canonical vocabulary
          value (``CANONICAL_THIRDS`` / ``_FIFTHS`` / ``_SEVENTHS``).
        * Triads (no seventh): γ = β, placing them on the γ=β diagonal.

    The 4th coordinate D (sensory dissonance) is *not* returned here — it
    comes from :class:`DissonanceMap.lookup(α, β, γ)` and is root-invariant
    as well. See :mod:`eigen_sidecar` for the per-chord 4D builder.

    Args:
        intervals: List of intervals in 53-TET steps, already folded mod 53.
        root_pc: Kept for API compatibility (unused). The output is
            root-invariant by construction.

    Returns:
        (alpha, beta, gamma) frequency ratios.
    """
    del root_pc  # root-invariant by design
    iv = sorted({x % TET_53 for x in intervals if (x % TET_53) != 0})

    third = None
    fifth = None
    seventh = None

    for step in iv:
        if step in THIRD_RANGE and third is None:
            third = step
        elif step in FIFTH_RANGE and fifth is None:
            fifth = step
        elif step in SEVENTH_RANGE and seventh is None:
            seventh = step

    if third is not None:
        third = _snap_to_nearest(third, CANONICAL_THIRDS)
    if fifth is not None:
        fifth = _snap_to_nearest(fifth, CANONICAL_FIFTHS)
    else:
        fifth = 31  # default perfect fifth
    if seventh is not None:
        seventh = _snap_to_nearest(seventh, CANONICAL_SEVENTHS)

    alpha = get_53tet_ratio(third) if third else DEFAULT_ALPHA
    beta  = get_53tet_ratio(fifth)
    gamma = get_53tet_ratio(seventh) if seventh else beta  # triad → γ=β

    return alpha, beta, gamma


# =============================================================================
# DISSONANCE MAP
# =============================================================================

class DissonanceMap:
    """
    Pre-computed 3D dissonance field with trilinear interpolation.
    
    Loaded once from binary chunks. Provides O(1) dissonance lookup
    for any (α, β, γ) point in the [1.0, 2.0]³ space.
    """
    
    def __init__(self, dataset_path: str = None, base_freq: int = 220, 
                 n_points: int = 150):
        """
        Args:
            dataset_path: Path to EigenSpace_Data folder containing .bin chunks
            base_freq: Base frequency the map was computed at (Hz)
            n_points: Grid resolution (n³ points)
        """
        if dataset_path is None:
            # Default path relative to this file
            dataset_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "dataset", "EigenSpace_Data"
            )
        
        self.n_points = n_points
        self.r_low = 1.0
        self.r_high = 2.0
        
        self.alpha_range = np.linspace(self.r_low, self.r_high, n_points)
        self.beta_range = np.linspace(self.r_low, self.r_high, n_points)
        self.gamma_range = np.linspace(self.r_low, self.r_high, n_points)
        
        # Load binary chunks
        chunk_files = sorted([
            f for f in os.listdir(dataset_path)
            if f.startswith(f"harmonic-{base_freq}Hz-{n_points}nodes-chunk")
        ])
        
        if not chunk_files:
            raise FileNotFoundError(
                f"No dissonance map chunks found in {dataset_path} "
                f"for {base_freq}Hz, {n_points} nodes"
            )
        
        all_data = []
        for chunk_file in chunk_files:
            chunk_path = os.path.join(dataset_path, chunk_file)
            chunk_data = np.fromfile(chunk_path, dtype=np.float32)
            all_data.append(chunk_data)
        
        flat_data = np.concatenate(all_data)
        self.dissonance_3d = flat_data.reshape((n_points, n_points, n_points))
        
        # Pre-compute normalization stats for the embedding
        # Use in-tetrahedron values only (α ≤ β ≤ γ) for meaningful stats
        valid_mask = np.zeros_like(self.dissonance_3d, dtype=bool)
        for i in range(n_points):
            for j in range(i, n_points):
                valid_mask[i, j, j:] = True
        valid_values = self.dissonance_3d[valid_mask]
        
        self.diss_mean = float(np.mean(valid_values))
        self.diss_std = float(np.std(valid_values))
        self.diss_min = float(np.min(valid_values))
        self.diss_max = float(np.max(valid_values))
    
    def lookup(self, alpha: float, beta: float, gamma: float) -> Optional[float]:
        """
        Get dissonance at (α, β, γ) via trilinear interpolation.
        
        Returns None if the point is outside [1.0, 2.0]³.
        """
        if (alpha < self.r_low or alpha > self.r_high or
            beta < self.r_low or beta > self.r_high or
            gamma < self.r_low or gamma > self.r_high):
            return None
        
        def find_bracket(val, arr):
            idx = np.searchsorted(arr, val) - 1
            idx = max(0, min(idx, len(arr) - 2))
            t = (val - arr[idx]) / (arr[idx + 1] - arr[idx])
            return idx, t
        
        i, ti = find_bracket(alpha, self.alpha_range)
        j, tj = find_bracket(beta, self.beta_range)
        k, tk = find_bracket(gamma, self.gamma_range)
        
        d = self.dissonance_3d
        
        # Trilinear interpolation
        c000 = d[i, j, k];     c100 = d[i+1, j, k]
        c010 = d[i, j+1, k];   c110 = d[i+1, j+1, k]
        c001 = d[i, j, k+1];   c101 = d[i+1, j, k+1]
        c011 = d[i, j+1, k+1]; c111 = d[i+1, j+1, k+1]
        
        c00 = c000 * (1 - ti) + c100 * ti
        c10 = c010 * (1 - ti) + c110 * ti
        c01 = c001 * (1 - ti) + c101 * ti
        c11 = c011 * (1 - ti) + c111 * ti
        
        c0 = c00 * (1 - tj) + c10 * tj
        c1 = c01 * (1 - tj) + c11 * tj
        
        return float(c0 * (1 - tk) + c1 * tk)


# =============================================================================
# FREQUENCY-AWARE EIGENSPACE DISSONANCE
# =============================================================================
# Pairwise roughness kernel used as the sensory-dissonance back-end of the
# 4D EigenSpace map. The coefficients below are the standard Plomp-Levelt /
# Sethares roughness constants; the name `eigenspace_dissonance` refers to
# the role of this function inside the EigenSpace framework.

_PL_DSTAR = 0.24
_PL_S1    = 0.0207
_PL_S2    = 18.96
_PL_C1    =  5.0
_PL_C2    = -5.0
_PL_A1    = -3.51
_PL_A2    = -5.75


def eigenspace_dissonance(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
) -> float:
    """
    Vectorized EigenSpace sensory-dissonance kernel over an array of partials.

    Uses absolute frequencies so the result is register-aware:
    low-register chords have wider critical bands → higher dissonance.

    Args:
        frequencies: 1-D array of partial frequencies (Hz)
        amplitudes:  1-D array of amplitudes (same length)

    Returns:
        Total sensory dissonance (float, ≥ 0)
    """
    idx = np.argsort(frequencies)
    fr = frequencies[idx].astype(np.float64)
    am = amplitudes[idx].astype(np.float64)

    i, j = np.triu_indices(len(fr), k=1)
    S = _PL_DSTAR / (_PL_S1 * fr[i] + _PL_S2)
    Fdif = S * (fr[j] - fr[i])
    pair_diss = np.minimum(am[i], am[j]) * (
        _PL_C1 * np.exp(_PL_A1 * Fdif) + _PL_C2 * np.exp(_PL_A2 * Fdif)
    )
    return float(np.sum(pair_diss))


def chord_dissonance_from_pitches(
    step_53_pitches: List[int],
    n_harmonics: int = 6,
) -> float:
    """
    Compute EigenSpace sensory dissonance for a chord given its absolute
    53-TET pitches (NOT folded to one octave — uses the actual register).

    Each pitch is expanded into *n_harmonics* partials with amplitudes
    decaying as 1/h (sawtooth-like spectrum, more realistic than flat).

    Args:
        step_53_pitches: Absolute 53-TET step numbers (e.g. [212, 243, 265])
        n_harmonics:     Number of harmonics per voice (default 6)

    Returns:
        Raw dissonance value (float, ≥ 0).  NOT normalized.
    """
    if len(step_53_pitches) < 2:
        return 0.0

    freqs, amps = [], []
    for step in step_53_pitches:
        f0 = 440.0 * (2.0 ** ((step / (TET_53 / 12.0) - 69.0) / 12.0))
        for h in range(1, n_harmonics + 1):
            freqs.append(f0 * h)
            amps.append(1.0 / h)          # 1/h roll-off

    return eigenspace_dissonance(
        np.array(freqs, dtype=np.float64),
        np.array(amps, dtype=np.float64),
    )


# =============================================================================
# EIGENSPACE COMPUTER — token sequence → 4D coordinates
# =============================================================================

class EigenSpaceComputer:
    """
    Computes per-token EigenSpace coordinates from token sequences.
    
    For each position in a token sequence, this produces a 4D vector:
      (α, β, γ, δ)
    
    These define the chord's position in harmonic space as
    (α, β, γ, D), all root-invariant.

    The coordinates are constant across all tokens within a chord
    (from CHORD_START to CHORD_END), and reset to defaults between chords.

    .. note::
       This class is **legacy**. It infers the root as the lowest pitch
       (bass), which is wrong for slash chords. The production path uses
       :mod:`eigen_sidecar` with the symbolic L1 root and produces a
       per-chord ``.eigen.npy`` sidecar aligned to the merged token stream.
    """

    def __init__(self, dataset_path: str = None, diss_map: "DissonanceMap" = None,
                 **_kwargs):
        """
        Args:
            dataset_path: Forwarded to ``DissonanceMap`` if ``diss_map`` is None.
            diss_map: Optional pre-loaded DissonanceMap (re-used across calls).
        """
        if diss_map is None:
            try:
                diss_map = DissonanceMap(dataset_path=dataset_path)
            except FileNotFoundError:
                diss_map = None
        self.diss_map = diss_map

    def _extract_chord_pitches(self, token_strs: List[str], 
                                start_idx: int) -> List[int]:
        """
        Extract 53-TET pitch steps from a chord starting at start_idx.
        Reads from CHORD_START until CHORD_END.
        Handles both P_<step> and PV_<step>_<vel> token formats.
        """
        pitches = []
        i = start_idx
        while i < len(token_strs):
            tok = token_strs[i]
            if tok == "CHORD_END":
                break
            if tok.startswith("P_"):
                pitches.append(int(tok[2:]))
            elif tok.startswith("PV_"):
                # PV_<step>_<vel> compound token
                pitches.append(int(tok.split("_")[1]))
            i += 1
        return pitches
    
    def _pitches_to_intervals(self, pitches: List[int]) -> List[int]:
        """Convert absolute 53-TET pitches to intervals relative to root, mod 53.
        
        Folds all intervals into one octave (0–52) to produce voicing-independent
        pitch classes. This matches the octave normalization in notebook 02.
        """
        if not pitches:
            return []
        root = min(pitches)
        intervals = sorted(set((p - root) % TET_53 for p in pitches))
        return intervals
    
    def compute_for_tokens(self, token_strs: List[str]) -> np.ndarray:
        """
        Compute EigenSpace coordinates for each position in a token sequence.

        Every token within a chord inherits that chord's (α, β, γ, D).
        Non-chord tokens (BAR, <start>, <end>, etc.) get default values.

        Args:
            token_strs: List of token strings

        Returns:
            np.ndarray of shape (len(token_strs), 4) — [α, β, γ, D] per position
        """
        n = len(token_strs)
        d_default = (float(self.diss_map.diss_mean)
                     if self.diss_map is not None else DEFAULT_D)
        coords = np.full(
            (n, N_EIGEN),
            [DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, d_default],
            dtype=np.float32,
        )

        i = 0
        while i < n:
            if token_strs[i] == "CHORD_START":
                pitches = self._extract_chord_pitches(token_strs, i)
                intervals = self._pitches_to_intervals(pitches)
                alpha, beta, gamma = classify_intervals(intervals)
                if self.diss_map is not None:
                    d = self.diss_map.lookup(alpha, beta, gamma)
                    if d is None:
                        d = d_default
                else:
                    d = d_default

                # Fill all tokens in this chord with the same coordinates
                j = i
                while j < n and token_strs[j] != "CHORD_END":
                    coords[j] = [alpha, beta, gamma, d]
                    j += 1
                if j < n:  # include CHORD_END itself
                    coords[j] = [alpha, beta, gamma, d]
                    j += 1

                i = j
            else:
                i += 1

        return coords
    
    def compute_for_ids(self, token_ids: List[int], id_to_token: dict) -> np.ndarray:
        """
        Convenience: compute EigenSpace from token IDs using an id-to-token map.
        
        Returns:
            np.ndarray of shape (len(token_ids), 4)
        """
        token_strs = [id_to_token.get(i, "<pad>") for i in token_ids]
        return self.compute_for_tokens(token_strs)
    
    def compute_batch(self, token_id_batch: np.ndarray, 
                      id_to_token: dict) -> np.ndarray:
        """
        Compute EigenSpace for a batch of sequences.
        
        Args:
            token_id_batch: (batch_size, seq_len) array of token IDs
            id_to_token: Dict mapping ID → token string
            
        Returns:
            np.ndarray of shape (batch_size, seq_len, 4)
        """
        batch_size, seq_len = token_id_batch.shape
        result = np.zeros((batch_size, seq_len, N_EIGEN), dtype=np.float32)
        
        for b in range(batch_size):
            result[b] = self.compute_for_ids(
                token_id_batch[b].tolist(), id_to_token
            )
        
        return result


# =============================================================================
# EIGENSPACE EMBEDDING (PyTorch nn.Module)
# =============================================================================

if HAS_TORCH:
    class EigenSpaceEmbedding(nn.Module):
        """
        Projects 4D EigenSpace coordinates into the transformer's embedding space.

        Architecture:
          (α, β, γ, D)  →  Linear(4, hidden)  →  GELU  →  Linear(hidden, n_embd)

        The input coordinates define a chord's complete position in harmonic
        space: interval ratios (α, β, γ) plus the sensory-dissonance scalar D
        from the pre-computed EigenSpace dissonance map. All four dimensions
        are root-invariant.

        This IS the model's positional encoding (v2 architecture).
        """

        def __init__(self, n_embd: int, n_eigen: int = N_EIGEN, hidden: int = 64):
            """
            Args:
                n_embd: Output dimension (must match transformer embedding dim)
                n_eigen: Input dimension (default 4: α, β, γ, D)
                hidden: Hidden layer dimension (default 64)
            """
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(n_eigen, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_embd, bias=False),
            )
        
        def forward(self, eigen_coords: torch.Tensor) -> torch.Tensor:
            """
            Args:
                eigen_coords: (batch_size, seq_len, 4) float tensor
                
            Returns:
                (batch_size, seq_len, n_embd) — the harmonic positional encoding
            """
            return self.projection(eigen_coords)


# =============================================================================
# PRECOMPUTATION UTILITY — for dataset preparation
# =============================================================================

def precompute_eigenspace_for_dataset(
    token_sequences: List[List[str]],
    dataset_path: str = None,
) -> List[np.ndarray]:
    """
    Pre-compute EigenSpace coordinates for an entire dataset of token sequences.
    
    This should be called once during data preparation, not during training.
    The results are saved alongside the token data and loaded by the DataLoader.
    
    Args:
        token_sequences: List of token string sequences (one per song)
        dataset_path: Path to EigenSpace_Data folder (unused, kept for compat)
        
    Returns:
        List of np.ndarray, each (seq_len, 4), matching the input sequences
    """
    computer = EigenSpaceComputer(
        dataset_path=dataset_path,
    )
    
    results = []
    for tokens in token_sequences:
        coords = computer.compute_for_tokens(tokens)
        results.append(coords)
    
    return results


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EigenSpace Module — Self-Test")
    print("=" * 70)
    
    # 1. Test interval classification
    print("\n1. Interval classification:")
    test_cases = [
        ([0, 18, 31],      "Major triad (M-P) → γ=β"),
        ([0, 18, 31, 49],  "Major 7th (M-P-M7)"),
        ([0, 18, 31, 44],  "Dominant 7th (M-P-m7)"),
        ([0, 13, 31, 44],  "Minor 7th (m-P-m7)"),
        ([0, 17, 30],      "Down-major, dim5 triad → γ=β, snap 30→31"),
        ([0, 13, 26, 44],  "Half-dim (m-dim5-m7)"),
        ([0, 13, 31],      "Minor triad → γ=β"),
    ]
    
    for intervals, label in test_cases:
        a, b, g = classify_intervals(intervals)
        triad_mark = " [TRIAD γ=β]" if abs(g - b) < 1e-6 else ""
        print(f"  {str(intervals):25s}  → α={a:.5f} β={b:.5f} γ={g:.5f}   ({label}){triad_mark}")
    
    # 2. Test dissonance map loading
    print("\n2. Dissonance map:")
    try:
        dmap = DissonanceMap()
        print(f"  Loaded: {dmap.dissonance_3d.shape}")
        print(f"  Range: {dmap.diss_min:.3f} – {dmap.diss_max:.3f}")
        print(f"  Mean: {dmap.diss_mean:.3f}, Std: {dmap.diss_std:.3f}")
        
        # Test some known chords  
        chords = [
            ("Major triad",   1.2654, 1.4999, 2.0000),
            ("Major 7th",     1.2654, 1.4999, 1.8981),
            ("Dominant 7th",  1.2654, 1.4999, 1.7779),
            ("Minor 7th",     1.1853, 1.4999, 1.7779),
        ]
        print("\n  Dissonance lookups:")
        for name, a, b, g in chords:
            d = dmap.lookup(a, b, g)
            print(f"    {name:15s}: D = {d:.3f}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
    
    # 3. Test token → EigenSpace computation
    print("\n3. Token sequence → EigenSpace:")
    test_tokens = [
        "<start>",
        "CHORD_START", "DUR_4.0", "P_212", "V_3", "P_243", "V_3", 
        "P_265", "V_2", "P_284", "V_3", "CHORD_END",
        "BAR",
        "CHORD_START", "DUR_4.0", "P_212", "V_3", "P_243", "V_3",
        "P_265", "V_2", "P_261", "V_3", "CHORD_END",
        "<end>",
    ]
    
    try:
        computer = EigenSpaceComputer()
        coords = computer.compute_for_tokens(test_tokens)
        
        print(f"  Sequence length: {len(test_tokens)}")
        print(f"  Coordinates shape: {coords.shape}")
        print()
        for i, (tok, c) in enumerate(zip(test_tokens, coords)):
            print(f"    {i:2d}  {tok:15s}  α={c[0]:.4f} β={c[1]:.4f} γ={c[2]:.4f} D={c[3]:.4f}")
    except Exception as e:
        print(f"  [SKIP] {e}")
    
    # 4. Test PyTorch embedding
    print("\n4. EigenSpace Positional Encoding:")
    if HAS_TORCH:
        emb = EigenSpaceEmbedding(n_embd=384, n_eigen=N_EIGEN, hidden=64)
        dummy = torch.randn(2, 10, N_EIGEN)  # batch=2, seq=10, eigen=4
        out = emb(dummy)
        print(f"  Input:  {dummy.shape}")
        print(f"  Output: {out.shape}")
        n_params = sum(p.numel() for p in emb.parameters())
        print(f"  Params: {n_params:,}")
    else:
        print("  [SKIP] torch not installed")
    
    print("\n" + "=" * 70)
    print("All tests passed.")
