"""
generate.py
===========
Generation script for the trained GPT-2 53-TET dual-channel model.

Loads a checkpoint from train.py and generates new microtonal
chord sequences using autoregressive sampling with live EigenSpace
recomputation — each newly generated chord gets its true (α, β, γ, D)
coordinates fed back into the model, rather than static defaults.

Generation modes
----------------
  1. Unconditional  — starts from <start> and generates freely
  2. Continuation   — provide a prompt sequence of tokens to continue from
  3. Interactive     — prompt from stdin, generate, show, repeat

Sampling strategies
-------------------
  --temperature    Softmax temperature (default 1.0)
  --top-k          Top-k filtering (default None)
  --top-p          Nucleus sampling threshold (default 0.95)

Usage
-----
  # Unconditional generation (256 new tokens):
  python generate.py

  # Use the best checkpoint:
  python generate.py --checkpoint ../checkpoints/best.pt

  # Control sampling:
  python generate.py --temperature 0.8 --top-k 50 --top-p 0.95

  # Generate more tokens:
  python generate.py --max-tokens 512

  # Generate multiple samples:
  python generate.py --num-samples 5

  # Prompt with specific tokens:
  python generate.py --prompt "<start> CHORD_START DUR_4.0"

  # Export to MIDI:
  python generate.py --midi output.mid

  # Interactive mode:
  python generate.py --interactive
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Path setup ──
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from eigenspace import EigenSpaceComputer


# =============================================================================
# Import model architecture from training script
# =============================================================================

# We import the classes directly from train to guarantee
# architecture parity with the checkpoint.
from importlib import util as _importlib_util

def _import_training_module():
    """Import train.py as a module."""
    spec = _importlib_util.spec_from_file_location(
        "train", _SRC_DIR / "train.py"
    )
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_train_mod = _import_training_module()
ModelConfig = _train_mod.ModelConfig
GPT2 = _train_mod.GPT2


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
    """Token <-> ID mapping loaded from vocab.json."""

    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            data = json.load(f)

        self.token_to_id: Dict[str, int] = data['token_to_id']
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.config = data.get('config', {})
        self.vocab_size = len(self.token_to_id)

        # Useful token IDs
        self.pad_id   = self.token_to_id.get('<pad>', 0)
        self.start_id = self.token_to_id.get('<start>', 1)
        self.end_id   = self.token_to_id.get('<end>', 2)
        self.sep_id   = self.token_to_id.get('<sep>', 3)
        self.chord_start_id = self.token_to_id.get('CHORD_START', 4)
        self.chord_end_id   = self.token_to_id.get('CHORD_END', 5)
        self.bar_id         = self.token_to_id.get('BAR', 6)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert token strings to IDs."""
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids) -> List[str]:
        """Convert token IDs to strings."""
        return [self.id_to_token.get(int(i), '<unk>') for i in ids]

    def __len__(self):
        return self.vocab_size


# =============================================================================
# EIGENSPACE-AWARE GENERATION
# =============================================================================

@torch.no_grad()
def generate_with_eigenspace(
    model: GPT2,
    vocab: Vocabulary,
    eigen_computer: EigenSpaceComputer,
    prompt_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.95,
    device: torch.device = torch.device('cpu'),
    stop_at_end: bool = True,
    recompute_interval: str = 'chord',
) -> Tuple[List[int], List[str]]:
    """
    Autoregressive generation with live EigenSpace recomputation.

    Instead of using static default eigenspace for generated tokens,
    this function recomputes (α, β, γ, D) whenever a new chord is
    completed (CHORD_END emitted), feeding the true harmonic coordinates
    back into the model for subsequent tokens.

    Args:
        model:              Trained GPT2 model (eval mode)
        vocab:              Vocabulary mapping
        eigen_computer:     EigenSpaceComputer instance
        prompt_ids:         Starting token IDs
        max_new_tokens:     Maximum tokens to generate
        temperature:        Sampling temperature (< 1 = more focused)
        top_k:              Top-k filtering (None = disabled)
        top_p:              Nucleus sampling (None = disabled)
        device:             Torch device
        stop_at_end:        Stop generation when <end> is produced
        recompute_interval: 'chord' (recompute after each chord) or
                            'bar' (recompute after each bar) or
                            'none' (use defaults like the basic generate)

    Returns:
        (token_ids, token_strings) — full sequence including prompt
    """
    model.eval()
    block_size = model.config.block_size

    # Build initial sequence
    all_ids = list(prompt_ids)

    # Compute eigenspace for the prompt
    all_tokens_str = vocab.decode(all_ids)
    eigen_array = eigen_computer.compute_for_tokens(all_tokens_str)

    # Track whether we're inside a chord (for recomputation)
    in_chord = all_tokens_str[-1] not in {'CHORD_END', 'BAR', '<start>', '<end>', '<sep>'}
    needs_recompute = False

    for step in range(max_new_tokens):
        # Crop to block_size
        seq_len = len(all_ids)
        if seq_len <= block_size:
            idx = torch.tensor([all_ids], dtype=torch.long, device=device)
            eigen = torch.tensor(
                eigen_array[np.newaxis, :, :], dtype=torch.float32, device=device
            )
        else:
            idx = torch.tensor(
                [all_ids[-block_size:]], dtype=torch.long, device=device
            )
            eigen = torch.tensor(
                eigen_array[np.newaxis, -block_size:, :],
                dtype=torch.float32, device=device,
            )

        # Forward pass (inference mode: only last position logits)
        logits, _ = model(idx, eigen=eigen)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = -float('Inf')
            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_token = vocab.id_to_token.get(next_id, '<unk>')

        # Append token
        all_ids.append(next_id)
        all_tokens_str.append(next_token)

        # ── EigenSpace update logic ──
        # Default eigenspace for non-chord tokens (matches training data)
        _DEFAULT_EIGEN = np.array([[1.0, 1.0, 2.0, 1.0]], dtype=np.float32)

        if recompute_interval == 'none':
            # Static defaults for every new token
            eigen_array = np.concatenate([eigen_array, _DEFAULT_EIGEN], axis=0)

        elif recompute_interval == 'chord':
            # Recompute full eigenspace only when a chord is complete.
            # During training, every token in a chord has that chord's
            # FULL eigenspace.  We can't know the chord identity until
            # CHORD_END, so we carry forward the last known eigenspace
            # for in-progress chords and recompute at CHORD_END — the
            # next forward pass then sees correct eigen for all completed
            # chords.
            if next_token == 'CHORD_END':
                # Chord just finished — recompute the whole sequence
                eigen_array = eigen_computer.compute_for_tokens(all_tokens_str)
            elif next_token in ('BAR', 'REST', '<start>', '<end>', '<sep>'):
                # Non-chord tokens get defaults (matching training)
                eigen_array = np.concatenate([eigen_array, _DEFAULT_EIGEN], axis=0)
            else:
                # Inside an in-progress chord — inherit last eigenspace
                if len(eigen_array) > 0:
                    eigen_array = np.concatenate(
                        [eigen_array, eigen_array[[-1]]], axis=0
                    )
                else:
                    eigen_array = np.concatenate(
                        [eigen_array, _DEFAULT_EIGEN], axis=0
                    )

        elif recompute_interval == 'bar':
            if next_token in ('BAR', 'CHORD_END'):
                eigen_array = eigen_computer.compute_for_tokens(all_tokens_str)
            else:
                if len(eigen_array) > 0:
                    eigen_array = np.concatenate(
                        [eigen_array, eigen_array[[-1]]], axis=0
                    )
                else:
                    eigen_array = np.concatenate(
                        [eigen_array, _DEFAULT_EIGEN], axis=0
                    )

        # Stop conditions
        if stop_at_end and next_id == vocab.end_id:
            break

    return all_ids, all_tokens_str


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_sequence(tokens: List[str], color: bool = True) -> str:
    """
    Pretty-print a token sequence with optional ANSI colors.

    Colors:
      - Structural (BAR, REST) → cyan
      - Chord boundaries → yellow
      - PV tokens (pitch+velocity) → green
      - ROOT tokens → blue
      - Duration tokens → magenta
      - Special → red
    """
    if not color:
        return ' '.join(tokens)

    RESET   = '\033[0m'
    CYAN    = '\033[36m'
    YELLOW  = '\033[33m'
    GREEN   = '\033[32m'
    MAGENTA = '\033[35m'
    BLUE    = '\033[34m'
    RED     = '\033[31m'
    BOLD    = '\033[1m'

    parts = []
    for tok in tokens:
        if tok in ('<start>', '<end>', '<sep>', '<pad>'):
            parts.append(f"{BOLD}{RED}{tok}{RESET}")
        elif tok in ('CHORD_START', 'CHORD_END'):
            parts.append(f"{YELLOW}{tok}{RESET}")
        elif tok in ('BAR', 'REST'):
            parts.append(f"{BOLD}{CYAN}{tok}{RESET}")
        elif tok.startswith('PV_'):
            parts.append(f"{GREEN}{tok}{RESET}")
        elif tok.startswith('ROOT_'):
            parts.append(f"{BLUE}{tok}{RESET}")
        elif tok.startswith('DUR_'):
            parts.append(f"{MAGENTA}{tok}{RESET}")
        else:
            parts.append(tok)

    return ' '.join(parts)


def format_as_readable(tokens: List[str]) -> str:
    """
    Convert token sequence into a compact readable format.

    Token format: CHORD_START DUR_4.0 ROOT_15 PV_212_3 PV_243_3 ... CHORD_END

    Example output:
      |  [root=15 | steps=212,243,265,284,306 | dur=4.0]
      |  [root=28 | steps=220,252,274,293 | dur=2.0]
    """
    lines = []
    current_line = ""

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == '<start>':
            lines.append("── START ──")
        elif tok == '<end>':
            if current_line:
                lines.append(current_line)
                current_line = ""
            lines.append("── END ──")
        elif tok == 'BAR':
            if current_line:
                lines.append(current_line)
                current_line = ""
            lines.append("  |")
        elif tok == 'CHORD_START':
            steps = []
            root = None
            dur = None
            i += 1
            while i < len(tokens) and tokens[i] != 'CHORD_END':
                t = tokens[i]
                if t.startswith('PV_'):
                    parts = t.split('_')   # ['PV', step, vel]
                    steps.append(parts[1])
                elif t.startswith('ROOT_'):
                    root = t[5:]           # strip 'ROOT_'
                elif t.startswith('DUR_'):
                    dur = t[4:]            # strip 'DUR_'
                i += 1
            root_str = f"root={root}" if root is not None else "root=?"
            steps_str = ','.join(steps) if steps else '—'
            dur_str = f"dur={dur}" if dur else "dur=?"
            current_line += f"  [{root_str} | steps={steps_str} | {dur_str}]"
        elif tok == 'REST':
            current_line += "  [REST]"

        i += 1

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)


def extract_chords_summary(tokens: List[str]) -> List[dict]:
    """
    Extract chord information from generated tokens.

    Token format per chord:
      CHORD_START DUR_4.0 ROOT_15 PV_212_3 PV_243_3 ... CHORD_END

    Returns a list of dicts with keys:
      - pitches:    list of absolute 53-TET step values (ints)
      - velocities: list of velocity bin values (ints, 1-8)
      - root:       root pitch class in 53-TET (int, 0-52), or None
      - duration:   float (beats), or None
      - bar_number: which bar the chord is in (int)
    """
    chords = []
    bar_num = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok == 'BAR':
            bar_num += 1
        elif tok == 'CHORD_START':
            chord = {
                'pitches': [],
                'velocities': [],
                'root': None,
                'duration': None,
                'bar_number': bar_num,
            }
            i += 1
            while i < len(tokens) and tokens[i] != 'CHORD_END':
                t = tokens[i]
                if t.startswith('PV_'):
                    parts = t.split('_')   # ['PV', step, vel]
                    chord['pitches'].append(int(parts[1]))
                    chord['velocities'].append(int(parts[2]))
                elif t.startswith('ROOT_'):
                    chord['root'] = int(t[5:])
                elif t.startswith('DUR_'):
                    chord['duration'] = float(t[4:])
                i += 1
            chords.append(chord)

        i += 1

    return chords


# =============================================================================
# MIDI EXPORT
# =============================================================================

def export_to_midi(
    tokens: List[str],
    output_path: str,
    tempo: int = 120,
    velocity: int = 80,
    pitch_offset: int = 106,
) -> str:
    """
    Export a generated token sequence to a MIDI file.

    Converts 53-TET pitches to 12-TET MIDI notes via rounding:
      midi_note = round(pitch_53tet * 12 / 53)

    If mido is available, creates a proper MIDI file.
    Otherwise falls back to a simpler format.

    Args:
        tokens:       List of token strings
        output_path:  Where to save the .mid file
        tempo:        BPM
        velocity:     MIDI velocity (0–127)
        pitch_offset: Offset subtracted to get 53-TET steps from pitch tokens

    Returns:
        Path to written file
    """
    try:
        from mido import MidiFile, MidiTrack, Message, MetaMessage
    except ImportError:
        print("  [WARNING] mido not installed — skipping MIDI export")
        print("  Install with: pip install mido")
        return ""

    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(track)

    # Tempo
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat))

    ticks_per_beat = mid.ticks_per_beat  # default 480

    chords = extract_chords_summary(tokens)

    for chord in chords:
        if not chord['pitches'] or chord['duration'] is None:
            continue

        dur_beats = chord['duration']
        dur_ticks = int(dur_beats * ticks_per_beat)

        # Convert 53-TET to 12-TET MIDI notes
        midi_notes = []
        for p53 in chord['pitches']:
            # Subtract offset to get 53-TET steps, convert to 12-TET
            step_53 = p53 - pitch_offset
            midi_note = round(step_53 * 12 / 53) + 36  # +36 for reasonable octave
            midi_note = max(0, min(127, midi_note))
            midi_notes.append(midi_note)

        # Remove duplicates, sort
        midi_notes = sorted(set(midi_notes))

        # Note-on events (simultaneous)
        for j, note in enumerate(midi_notes):
            track.append(Message('note_on', note=note, velocity=velocity, time=0))

        # Note-off events
        for j, note in enumerate(midi_notes):
            track.append(Message(
                'note_off', note=note, velocity=0,
                time=dur_ticks if j == 0 else 0,
            ))

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mid.save(output_path)
    return output_path


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    vocab_size: int = None,
) -> Tuple[GPT2, dict]:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to .pt file
        device: Target device
        vocab_size: Override vocab size (if not in checkpoint)

    Returns:
        (model, checkpoint_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    cfg = checkpoint['model_config']
    model_config = ModelConfig(
        vocab_size=cfg.get('vocab_size', vocab_size or 345),
        block_size=cfg.get('block_size', 1024),
        n_layer=cfg.get('n_layer', 6),
        n_head=cfg.get('n_head', 6),
        n_embd=cfg.get('n_embd', 384),
        dropout=cfg.get('dropout', 0.0),  # No dropout at inference
        bias=cfg.get('bias', False),
        use_eigenspace=cfg.get('use_eigenspace', True),
        n_eigen=cfg.get('n_eigen', 3),
        eigen_hidden=cfg.get('eigen_hidden', 64),
        max_local_pos=cfg.get('max_local_pos', 20),
        use_sequential_pos=cfg.get('use_sequential_pos', False),
    )

    # Build model
    model = GPT2(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    iter_num = checkpoint.get('iter_num', '?')
    best_val = checkpoint.get('best_val_loss', '?')
    print(f"  Loaded at iter {iter_num}, best_val_loss={best_val}")
    print(f"  Config: L={model_config.n_layer} H={model_config.n_head} "
          f"E={model_config.n_embd} eigen={model_config.use_eigenspace}")

    return model, checkpoint


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 53-TET chord sequences with trained GPT-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py
  python generate.py --checkpoint ../checkpoints/best.pt --temperature 0.8
  python generate.py --max-tokens 512 --top-k 50 --num-samples 3
  python generate.py --prompt "<start> BAR CHORD_START DUR_4.0"
  python generate.py --midi output.mid
  python generate.py --interactive
        """
    )

    # Checkpoint & data
    parser.add_argument("--checkpoint", type=str,
                        default=str(_ROOT_DIR / "checkpoints" / "best.pt"),
                        help="Path to model checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--vocab", type=str,
                        default=str(_ROOT_DIR / "dataset" / "tokenized" / "vocab.json"),
                        help="Path to vocab.json")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max new tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k filtering (default: None = disabled)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling p (default: 0.95)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None = random)")

    # Prompt
    parser.add_argument("--prompt", type=str, default=None,
                        help='Prompt tokens (space-separated, e.g. "<start> BAR")')

    # EigenSpace
    parser.add_argument("--eigen-mode", type=str, default="chord",
                        choices=["chord", "bar", "none"],
                        help="EigenSpace recomputation: chord|bar|none (default: chord)")

    # Output
    parser.add_argument("--midi", type=str, default=None,
                        help="Export to MIDI file path")
    parser.add_argument("--midi-tempo", type=int, default=120,
                        help="MIDI export tempo (BPM)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored terminal output")
    parser.add_argument("--format", type=str, default="both",
                        choices=["raw", "readable", "both"],
                        help="Output format (default: both)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Save generated tokens as JSON")

    # Mode
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive generation mode")

    # System
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto|cuda|cpu")

    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Seed ──
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}")

    # ── Load vocab ──
    vocab = Vocabulary(args.vocab)
    print(f"Vocabulary: {len(vocab)} tokens")

    # ── Load model ──
    model, ckpt = load_checkpoint(args.checkpoint, device, vocab.vocab_size)

    # ── EigenSpace computer ──
    print("Loading EigenSpace dissonance map...")
    eigen_computer = EigenSpaceComputer(normalize_diss=True)
    print("  EigenSpace ready")

    # ── Build prompt ──
    if args.prompt:
        prompt_tokens = args.prompt.strip().split()
        # Validate tokens
        unknown = [t for t in prompt_tokens if t not in vocab.token_to_id]
        if unknown:
            print(f"\n[WARNING] Unknown tokens in prompt: {unknown}")
            print(f"  Available special tokens: <start>, <end>, <sep>, BAR, REST,")
            print(f"  CHORD_START, CHORD_END, DUR_0.5..DUR_16.0, P_106..P_424, V_1..V_8")
            prompt_tokens = [t for t in prompt_tokens if t in vocab.token_to_id]

        prompt_ids = vocab.encode(prompt_tokens)
    else:
        # Default: start with <start>
        prompt_tokens = ['<start>']
        prompt_ids = [vocab.start_id]

    print(f"\nPrompt ({len(prompt_ids)} tokens): {' '.join(prompt_tokens)}")

    # ── Interactive mode ──
    if args.interactive:
        print(f"\n{'='*60}")
        print("Interactive Generation Mode")
        print(f"{'='*60}")
        print("Commands:")
        print("  Enter prompt tokens (space-separated) or press Enter for unconditional")
        print("  Type 'quit' or 'exit' to stop")
        print("  Type 'config' to view/change settings")
        print(f"{'='*60}\n")

        temperature = args.temperature
        top_k = args.top_k
        top_p = args.top_p
        max_tokens = args.max_tokens

        while True:
            try:
                user_input = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting.")
                break

            if user_input.lower() in ('quit', 'exit', 'q'):
                break

            if user_input.lower() == 'config':
                print(f"  temperature = {temperature}")
                print(f"  top_k       = {top_k}")
                print(f"  top_p       = {top_p}")
                print(f"  max_tokens  = {max_tokens}")
                try:
                    change = input("  Change? (e.g. 'temperature 0.8' or Enter to skip): ").strip()
                    if change:
                        key, val = change.split(None, 1)
                        if key == 'temperature':
                            temperature = float(val)
                        elif key == 'top_k':
                            top_k = int(val) if val != 'None' else None
                        elif key == 'top_p':
                            top_p = float(val) if val != 'None' else None
                        elif key == 'max_tokens':
                            max_tokens = int(val)
                        print(f"  Updated {key} = {val}")
                except (ValueError, KeyboardInterrupt):
                    pass
                continue

            # Parse prompt
            if user_input:
                p_tokens = user_input.split()
                unknown = [t for t in p_tokens if t not in vocab.token_to_id]
                if unknown:
                    print(f"  [WARNING] Unknown tokens: {unknown}")
                    p_tokens = [t for t in p_tokens if t in vocab.token_to_id]
                p_ids = vocab.encode(p_tokens)
            else:
                p_tokens = ['<start>']
                p_ids = [vocab.start_id]

            # Generate
            t0 = time.time()
            gen_ids, gen_tokens = generate_with_eigenspace(
                model, vocab, eigen_computer,
                prompt_ids=p_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                recompute_interval=args.eigen_mode,
            )
            dt = time.time() - t0

            print(f"\nGenerated {len(gen_ids) - len(p_ids)} tokens in {dt:.2f}s "
                  f"({(len(gen_ids) - len(p_ids))/dt:.0f} tok/s)")
            print()
            print(format_sequence(gen_tokens, color=not args.no_color))
            print()
            print(format_as_readable(gen_tokens))

        return

    # ── Batch generation ──
    print(f"\n{'='*60}")
    print(f"Generating {args.num_samples} sample(s)")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Top-k:         {args.top_k}")
    print(f"  Top-p:         {args.top_p}")
    print(f"  EigenSpace:    {args.eigen_mode}")
    print(f"{'='*60}")

    all_results = []

    for sample_idx in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n── Sample {sample_idx + 1}/{args.num_samples} ──")

        t0 = time.time()
        gen_ids, gen_tokens = generate_with_eigenspace(
            model, vocab, eigen_computer,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            recompute_interval=args.eigen_mode,
        )
        dt = time.time() - t0
        new_tokens = len(gen_ids) - len(prompt_ids)

        print(f"\nGenerated {new_tokens} tokens in {dt:.2f}s "
              f"({new_tokens/dt:.0f} tok/s)")

        # Display
        if args.format in ('raw', 'both'):
            print(f"\n[Raw]\n{format_sequence(gen_tokens, color=not args.no_color)}")

        if args.format in ('readable', 'both'):
            print(f"\n[Readable]")
            print(format_as_readable(gen_tokens))

        # Chord summary
        chords = extract_chords_summary(gen_tokens)
        n_bars = sum(1 for t in gen_tokens if t == 'BAR')
        print(f"\n  Summary: {len(chords)} chords across {n_bars} bars")

        all_results.append({
            'token_ids': gen_ids,
            'token_strings': gen_tokens,
            'n_chords': len(chords),
            'n_bars': n_bars,
            'generation_time': dt,
        })

        # MIDI export
        if args.midi:
            midi_path = args.midi
            if args.num_samples > 1:
                base, ext = os.path.splitext(args.midi)
                midi_path = f"{base}_{sample_idx + 1}{ext}"

            path = export_to_midi(
                gen_tokens, midi_path, tempo=args.midi_tempo,
                pitch_offset=vocab.config.get('pitch_offset', 106),
            )
            if path:
                print(f"  MIDI exported: {path}")

    # JSON export
    if args.json_out:
        json_data = []
        for r in all_results:
            json_data.append({
                'tokens': r['token_strings'],
                'token_ids': r['token_ids'],
                'n_chords': r['n_chords'],
                'n_bars': r['n_bars'],
            })
        with open(args.json_out, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nJSON saved: {args.json_out}")

    print(f"\n{'='*60}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
