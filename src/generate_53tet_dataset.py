"""
Script to generate 53-TET MIDI and Text files in parallel.
Reads 12-TET MIDI files from dataset/midi_files/12_tet_mpe
Outputs 53-TET MIDI files to dataset/midi_files/53_tet_mpe/type_<label>/<stem>_<type>.mid
Outputs 53-TET Text files to dataset/text_files/53_tet_mpe/type_<label>/<stem>_<type>.txt
(Text tree is a 1:1 mirror of the MIDI tree.)
"""

import os
import sys
import ast
import traceback
import argparse
import mido
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Ensure local imports work whether run from src or root
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    import convention
except ImportError:
    print("⚠️ Could not import convention module. Ensure it exists in src/")

try:
    import chord_mapping
except ImportError:
    print("⚠️ Could not import chord_mapping module.")

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

# 53-TET Note Definitions (0-52)
NOTE_NAMES_53TET = [
    "C", "^C", "^^C", "vvC#", "vC#", "C#", "^C#", "^^C#", "vD", "D", 
    "^D", "^^D", "vvD#", "vD#", "D#", "^^Eb", "vvE", "vE", "E", "^E", 
    "^^E", "vF", "F", "^F", "^^F", "vvF#", "vF#", "F#", "^F#", "^^F#", 
    "vG", "G", "^G", "^^G", "vvG#", "vG#", "G#", "^G#", "vvA", "vA", 
    "A", "^A", "^^A", "vBb", "Bb", "^Bb", "^^Bb", "vvB", "vB", "B", 
    "^B", "^^B", "vC"
]

MODAL_SCALE_TYPES = {
    # Type 0: Major (Standard 12-TET)
    'type_0_major': {
        'name': 'Major',
        'hc_distances': [0, 9, 9, 4, 9, 9, 9],
        'description': 'Standard major scale (baseline 12-TET)'
    },
    'type_0_minor': {
        'name': 'Major_Minor',
        'hc_distances': [0, 9, 4, 9, 9, 4, 9],
        'description': 'Standard major (Minor rotation)'
    },

    # Type 1: Neutral
    'type_1_neutral': {
        'name': 'Neutral',
        'hc_distances': [0, 8, 7, 7, 9, 8, 7],
        'description': 'Neutral mode with neutral intervals'
    },
    'type_1_minor': {
        'name': 'Neutral_Minor',
        'hc_distances': [0, 7, 7, 8, 7, 7, 9],
        'description': 'Neutral mode (Minor rotation)'
    },

    # Type 2: SubMinor
    'type_2_subminor': {
        'name': 'SubMinor',
        'hc_distances': [0, 9, 3, 10, 9, 3, 10],
        'description': 'Subminor mode'
    },
    'type_2_minor': {
        'name': 'SubMinor_Minor',
        'hc_distances': [0, 10, 9, 9, 3, 10, 9],
        'description': 'Subminor mode (Minor rotation)'
    },

    # Type 3: H_3rd_H_7th
    'type_3_major': {
        'name': 'H_3rd_H_7th',
        'hc_distances': [0, 9, 8, 5, 9, 8, 5],
        'description': 'Harmonic 3rd and 7th mode'
    },
    'type_3_minor': {
        'name': 'H_3rd_H_7th_Minor',
        'hc_distances': [0, 5, 9, 9, 8, 5, 9],
        'description': 'Harmonic 3rd and 7th mode (Minor rotation)'
    },

    # Type 4: UpMajor
    'type_4_upmajor': {
        'name': 'UpMajor',
        'hc_distances': [0, 10, 9, 3, 9, 10, 9],
        'description': 'Up-major mode'
    },
    'type_4_minor': {
        'name': 'UpMajor_Minor',
        'hc_distances': [0, 9, 3, 10, 9, 3, 9],
        'description': 'Up-major mode (Minor rotation)'
    },

    # Type 5: Major_v2
    'type_5_major_v2': {
        'name': 'Major_v2',
        'hc_distances': [0, 9, 9, 5, 8, 8, 9],
        'description': 'Alternative major mode'
    },
    'type_5_minor': {
        'name': 'Major_v2_Minor',
        'hc_distances': [0, 9, 5, 9, 9, 5, 8],
        'description': 'Alternative major mode (Minor rotation)'
    },

    # Type 6: Neutral_N
    'type_6_neutral_n': {
        'name': 'Neutral_N',
        'hc_distances': [0, 8, 7, 7, 9, 5, 10],
        'description': 'Neutral N mode'
    },
    'type_6_minor': {
        'name': 'Neutral_N_Minor',
        'hc_distances': [0, 10, 7, 8, 7, 7, 9],
        'description': 'Neutral N mode (Minor rotation)'
    }
}

# ==========================================
# LOGIC FUNCTIONS
# ==========================================

def identify_semantic_quality(steps):
    if steps is None: return None
    steps = steps % 53
    if 'convention' in sys.modules:
        return convention.STEP_TO_SEMANTIC.get(steps, f"step{steps}")
    return f"step{steps}"

def get_new_chord_name(intervals_in_steps):
    if 'convention' in sys.modules:
        q3 = identify_semantic_quality(intervals_in_steps.get('third'))
        q5 = identify_semantic_quality(intervals_in_steps.get('fifth'))
        q7 = identify_semantic_quality(intervals_in_steps.get('seventh'))
        return convention.get_name(q3, q5, q7)
    return str(intervals_in_steps)

def get_53tet_ratio(steps):
    return 2 ** (steps / 53.0)

def find_closest_53tet_step(ratio):
    steps = round(53 * np.log2(ratio))
    return steps

def build_chromatic_scale_53tet(hc_distances, root_step=0, tonic_position=0, is_minor=False):
    scale_7_steps = np.cumsum(hc_distances)
    
    # Derive chromatic positions from the interval sizes themselves.
    # Each cumulative step maps to its nearest 12-TET position via round(step * 12/53).
    # This correctly handles all scale types without relying on is_minor.
    relative_positions = [round(step * 12 / 53) for step in scale_7_steps]
    
    # Build scale with root at position 0 (key-independent)
    chromatic_from_root = [None] * 12
    for i, pos in enumerate(relative_positions):
        if pos < 12:
            chromatic_from_root[pos] = int(scale_7_steps[i])
    
    # Interpolate the 5 non-scale chromatic positions
    for i in range(12):
        if chromatic_from_root[i] is None:
            prev_pos = -1
            next_pos = 12
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
                range_steps = next_step - prev_step
                positions = next_pos - prev_pos
                offset = i - prev_pos
                chromatic_from_root[i] = round(prev_step + (range_steps * offset / positions))
            elif prev_pos >= 0:
                prev_step = chromatic_from_root[prev_pos]
                range_steps = 53 - prev_step
                positions = 12 - prev_pos
                offset = i - prev_pos
                chromatic_from_root[i] = round(prev_step + (range_steps * offset / positions))
            else:
                chromatic_from_root[i] = round((i / 12) * 53)
    
    # Rotate to the actual key: shift by tonic's 53-TET position
    tonic_53tet = round(tonic_position * 53 / 12)
    chromatic_steps = [
        (tonic_53tet + chromatic_from_root[(j - tonic_position) % 12]) % 53
        for j in range(12)
    ]
    return chromatic_steps

def calculate_53tet_frequency(midi_note, chromatic_scale_steps):
    tet12_freq = 440.0 * (2 ** ((midi_note - 69) / 12))
    note_in_octave = midi_note % 12
    step_53tet = chromatic_scale_steps[note_in_octave]
    step_12tet = (note_in_octave * 53) / 12
    hc_deviation = step_53tet - step_12tet
    # Handle wrap-around at octave boundary (e.g. step 52 vs neutral 0)
    if hc_deviation > 26.5:
        hc_deviation -= 53
    elif hc_deviation < -26.5:
        hc_deviation += 53
    ratio = 2 ** (hc_deviation / 53)
    frequency = tet12_freq * ratio
    return frequency

def calculate_pitch_bend_for_frequency(target_freq, midi_note):
    a4_freq = 440.0
    a4_midi = 69
    tet12_freq = a4_freq * (2 ** ((midi_note - a4_midi) / 12))
    if tet12_freq > 0:
        cents = 1200 * np.log2(target_freq / tet12_freq)
    else:
        cents = 0
    return cents

def process_text_file_conversion(input_path, scale_type, key, chromatic_scale, output_dir=None):
    """
    Finds the corresponding text file for a MIDI file and converts its chords to 53-TET notation.
    """
    # Locate text directory
    potential_text_dirs = [
        input_path.parent.parent.parent / "text_files" / "12_tet_files",
        input_path.parent.parent.parent / "text_files", 
        input_path.parent.parent / "text_files",        
        Path("dataset/text_files/12_tet_files").resolve(),
        Path("dataset/text_files").resolve(),
        Path("../dataset/text_files/12_tet_files").resolve(),
        Path("../dataset/text_files").resolve()
    ]
    
    text_dir = None
    for d in potential_text_dirs:
        if d.exists():
            # Check if file actually exists in this dir to avoid false positives with empty dirs
            if (d / (input_path.stem + ".txt")).exists():
                text_dir = d
                break
    
    # If not found by specific file check, fall back to first existing dir (legacy behavior)
    if text_dir is None:
        for d in potential_text_dirs:
            if d.exists():
                text_dir = d
                break
            
    if text_dir is None:
        # Fallback to checking typical location relative to script
        script_dataset_text = Path(__file__).parent.parent / "dataset" / "text_files"
        if script_dataset_text.exists():
             text_dir = script_dataset_text

    if text_dir is None:
        print(f"⚠️ Could not locate text_files directory for {input_path.name}")
        return

    text_filename = input_path.stem + ".txt"
    text_path = text_dir / text_filename
    
    if not text_path.exists():
        # Try finding without some suffices if needed
        text_filename_alt = input_path.stem.split("_type")[0] + ".txt" 
        text_path_alt = text_dir / text_filename_alt
        if text_path_alt.exists():
             text_path = text_path_alt
        else:
            # print(f"ℹ️ Corresponding text file not found: {text_path}")
            return
        
    try:
        with open(text_path, 'r') as f:
            content = f.read()
            try:
                chord_data = ast.literal_eval(content)
            except:
                print(f"Could not parse text file as list structure: {text_path.name}")
                return
            
        modified_chords = []
        
        chromatic_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        i = 0
        while i < len(chord_data):
            token = chord_data[i]
            
            # Handle Slash chords (Bass note) 
            if token == '/':
                modified_chords.append(token)
                i += 1
                if i < len(chord_data):
                    bass_token = chord_data[i]
                    if bass_token in chromatic_map:
                         bass_idx = chromatic_map[bass_token]
                         bass_step = chromatic_scale[bass_idx]
                         new_bass_name = NOTE_NAMES_53TET[bass_step % 53]
                         modified_chords.append(new_bass_name)
                    else:
                         modified_chords.append(bass_token) 
                    i += 1
                continue

            # Check for Root match
            is_root = False
            root_text = ""
            
            if token in chromatic_map:
                is_root = True
                root_text = token
            
            if not is_root:
                modified_chords.append(token)
                i += 1
                continue
            
            # Found Root
            quality_text = ""
            has_quality_token = False
            current_idx = i
            
            if current_idx + 1 < len(chord_data):
                next_tok = chord_data[current_idx + 1]
                if (next_tok not in ['|', '|:', ':|', 'e||', 'b||', '/', '.'] 
                    and not next_tok.startswith('Form_')
                    and not (next_tok.replace('.','',1).isdigit())): 
                    
                    quality_text = next_tok
                    has_quality_token = True
            
            # Parse Intervals
            input_intervals = {
                'third': 4, 
                'fifth': 7, 
                'seventh': None 
            }
            
            q = quality_text
            if 'maj7' in q or 'Maj7' in q:
                input_intervals['seventh'] = 11
            elif 'maj' in q: 
                pass
            elif 'dom7' in q or q == '7':
                input_intervals['seventh'] = 10
            elif 'm7' in q or 'min7' in q: 
                input_intervals['third'] = 3
                input_intervals['seventh'] = 10
            elif 'm' in q or 'min' in q:
                input_intervals['third'] = 3
                if '7' in q: input_intervals['seventh'] = 10 
            elif 'dim' in q or 'ø' in q:
                input_intervals['third'] = 3
                input_intervals['fifth'] = 6
                if '7' in q: input_intervals['seventh'] = 9 
                if 'ø' in q: input_intervals['seventh'] = 10 
            elif 'aug' in q or '+' in q:
                input_intervals['fifth'] = 8
                if '7' in q: input_intervals['seventh'] = 10
                
            # Calculate 53-TET steps
            root_idx_12 = chromatic_map[root_text]
            root_step = chromatic_scale[root_idx_12]
            
            def get_step_from_12tet_interval(root_step, interval_12):
                target_idx_12 = (root_idx_12 + interval_12) % 12
                target_step = chromatic_scale[target_idx_12]
                diff = target_step - root_step
                if diff < 0: diff += 53
                return diff

            steps_map = {}
            steps_map['third'] = get_step_from_12tet_interval(root_step, input_intervals['third'])
            steps_map['fifth'] = get_step_from_12tet_interval(root_step, input_intervals['fifth'])
            
            if input_intervals['seventh'] is not None:
                steps_map['seventh'] = get_step_from_12tet_interval(root_step, input_intervals['seventh'])
            else:
                steps_map['seventh'] = None
                
            # Get New Suffix
            new_suffix = get_new_chord_name(steps_map)
            
            new_root_name = NOTE_NAMES_53TET[root_step % 53]
            modified_chords.append(new_root_name)
            modified_chords.append(new_suffix)
            
            if has_quality_token:
                i += 2 
            else:
                i += 1 

        if output_dir is None:
             output_path_dir = input_path.parent
        else:
             output_path_dir = output_dir

        output_text_filename = f"{input_path.stem}_{scale_type}.txt"
        output_text_path = output_path_dir / output_text_filename
        
        with open(output_text_path, 'w') as f:
            f.write(str(modified_chords))
            
    except Exception as e:
        print(f"❌ Error processing text file {text_path.name}: {e}")
        # traceback.print_exc()

def convert_midi_to_53tet(input_midi_path, scale_type='type_1', output_dir=None, text_output_dir=None, key=None):
    
    input_path = Path(input_midi_path)
    if scale_type not in MODAL_SCALE_TYPES:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    if key is None:
        import re
        # Case insensitive match for key in filename (e.g. Name_Key_Tonality)
        match = re.search(r'_([a-zA-Z#]+)_(major|minor)$', input_path.stem, re.IGNORECASE)
        if match:
            key = match.group(1)
            tonality_str = match.group(2).lower()
            # Normalize key case just in case (e.g. 'bb' -> 'Bb')
            if len(key) > 1:
                key = key[0].upper() + key[1].lower()
            else:
                key = key.upper()
        else:
            key = 'C'
            tonality_str = 'major'
            # print(f"⚠️  Could not detect key from filename, defaulting to C")
    else:
        tonality_str = 'major' # Default if key manually provided without tonality context

    is_minor = 'minor' in tonality_str
    
    key_to_position = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    tonic_position = key_to_position.get(key, 0)
    config = MODAL_SCALE_TYPES[scale_type]
    
    # Just use the defined distances for this specific type
    hc_distances = config['hc_distances']
    
    chromatic_scale = build_chromatic_scale_53tet(hc_distances, root_step=0, tonic_position=tonic_position, is_minor=is_minor)
    
    mid = mido.MidiFile(input_path)
    # Preserve original track structure (matching 04_53TET_conversion.ipynb approach)
    out_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    
    active_notes = {}  # note -> channel
    next_channel = 1   # Channels 1-15 for MPE notes, 0 for global
    
    for track in mid.tracks:
        out_track = mido.MidiTrack()
        out_mid.tracks.append(out_track)
        
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Allocate MPE channel for this note
                channel = next_channel
                next_channel = (next_channel % 15) + 1
                active_notes[msg.note] = channel
                
                # Calculate 53-TET pitch bend
                target_freq = calculate_53tet_frequency(msg.note, chromatic_scale)
                bend_cents = calculate_pitch_bend_for_frequency(target_freq, msg.note)
                bend_value = int((bend_cents / 200) * 8192)
                bend_value = max(-8192, min(8191, bend_value))
                
                if bend_value != 0:
                    # Pitch bend before note_on
                    out_track.append(mido.Message('pitchwheel', channel=channel, pitch=bend_value, time=msg.time))
                    out_track.append(mido.Message('note_on', note=msg.note, velocity=msg.velocity, channel=channel, time=0))
                else:
                    out_track.append(mido.Message('note_on', note=msg.note, velocity=msg.velocity, channel=channel, time=msg.time))
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                channel = active_notes.pop(msg.note, msg.channel)
                out_track.append(mido.Message('note_off', note=msg.note, velocity=msg.velocity if msg.type == 'note_off' else 0, channel=channel, time=msg.time))
                # Reset pitch bend on this channel
                out_track.append(mido.Message('pitchwheel', channel=channel, pitch=0, time=0))
            
            elif msg.type in ['program_change', 'control_change']:
                out_track.append(msg.copy())
            
            else:
                # Meta messages and everything else: copy as-is
                out_track.append(msg.copy())
    
    mpe_midi = out_mid
    
    # Process corresponding Text file
    # If text_output_dir is provided, use it, else use MIDI output dir
    eff_text_dir = text_output_dir if text_output_dir else output_dir
    process_text_file_conversion(input_path, scale_type, key, chromatic_scale, eff_text_dir)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{input_path.stem}_{scale_type}.mid"
    output_path = output_dir / output_filename
    
    mpe_midi.save(output_path)
    return output_path

# ==========================================
# PARALLEL EXECUTION
# ==========================================

def process_single_file(args):
    """
    Wrapper for parallel execution
    """
    midi_path, output_midi_dir, output_text_dir, scale_type = args
    try:
        convert_midi_to_53tet(midi_path, scale_type=scale_type, output_dir=output_midi_dir, text_output_dir=output_text_dir)
        return True, midi_path
    except Exception as e:
        # traceback.print_exc()
        return False, f"{midi_path.name}: {str(e)}"

def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Generate 53-TET dataset from 12-TET MIDI files.")
    parser.add_argument('--types', nargs='+', help='List of scale types to generate (e.g., type_0 type_1). If not provided, generates all.')
    parser.add_argument('--workers', type=int, default=6, help='Number of parallel workers (default: 6).')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of input MIDI files to process (0 = all). Useful for testing.')
    args = parser.parse_args()

    # Helper to resolve paths relative to this script or workspace
    ROOT_DIR = Path(__file__).parent.parent
    
    # Define directories
    # INPUT: dataset/midi_files/12_tet_mpe
    INPUT_MIDI_DIR = ROOT_DIR / 'dataset' / 'midi_files' / '12_tet_mpe'
    
    # OUTPUT: dataset/midi_files/53_tet_mpe (organized by type subfolders)
    OUTPUT_MIDI_BASE = ROOT_DIR / 'dataset' / 'midi_files' / '53_tet_mpe'
    
    # OUTPUT TEXT: dataset/text_files/53_tet_mpe (organized by type subfolders — mirror of MIDI tree)
    OUTPUT_TEXT_BASE = ROOT_DIR / 'dataset' / 'text_files' / '53_tet_mpe'
    
    # Configuration
    NUM_WORKERS = args.workers
    
    # Determine Scale Types
    if args.types:
        # Validate provided types
        TARGET_SCALE_TYPES = []
        for t in args.types:
            if t in MODAL_SCALE_TYPES:
                TARGET_SCALE_TYPES.append(t)
            else:
                print(f"⚠️ Warning: '{t}' is not a valid scale type. Skipping.")
        
        if not TARGET_SCALE_TYPES:
            print("❌ No valid scale types selected. Exiting.")
            print(f"Available types: {list(MODAL_SCALE_TYPES.keys())}")
            return
    else:
        # Generate for all available scale types defined in MODAL_SCALE_TYPES
        TARGET_SCALE_TYPES = list(MODAL_SCALE_TYPES.keys())
    
    LIMIT = args.limit
    
    print(f"--- 53-TET DATASET GENERATOR ---")
    print(f"Input Directory: {INPUT_MIDI_DIR}")
    print(f"MIDI Output:     {OUTPUT_MIDI_BASE}")
    print(f"Text Output:     {OUTPUT_TEXT_BASE}")
    print(f"Scale Types:     {TARGET_SCALE_TYPES}")
    print(f"Workers:         {NUM_WORKERS}")
    print(f"Limit:           {LIMIT if LIMIT > 0 else 'ALL'}")
    
    # Validate Inputs
    if not INPUT_MIDI_DIR.exists():
        print(f"❌ Input directory does not exist: {INPUT_MIDI_DIR}")
        return

    # Create Outputs
    OUTPUT_MIDI_BASE.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEXT_BASE.mkdir(parents=True, exist_ok=True)
    
    # Collect Files
    midi_files = sorted(INPUT_MIDI_DIR.glob('*.mid'))
    print(f"Found {len(midi_files)} MIDI files in source.")
    
    if LIMIT > 0:
        midi_files = midi_files[:LIMIT]
        print(f"Limited to {LIMIT} files for testing.")
    
    if not midi_files:
        print("No MIDI files found.")
        return

    # Prepare Tasks — each scale type gets its own subfolder in BOTH trees
    tasks = []
    for scale_type in TARGET_SCALE_TYPES:
        type_midi_dir = OUTPUT_MIDI_BASE / scale_type
        type_text_dir = OUTPUT_TEXT_BASE / scale_type
        type_midi_dir.mkdir(parents=True, exist_ok=True)
        type_text_dir.mkdir(parents=True, exist_ok=True)
    for m in midi_files:
        for scale_type in TARGET_SCALE_TYPES:
            type_midi_dir = OUTPUT_MIDI_BASE / scale_type
            type_text_dir = OUTPUT_TEXT_BASE / scale_type
            tasks.append((m, type_midi_dir, type_text_dir, scale_type))
    
    # Execute Parallel
    print("\nStarting parallel processing...")
    success_count = 0
    failures = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), unit="file"))
        
        for success, msg in results:
            if success:
                success_count += 1
            else:
                failures.append(msg)
    
    print(f"\n✅ Completed.")
    print(f"Success: {success_count}/{len(tasks)}")
    print(f"Failed:  {len(failures)}")
    
    if failures:
        print("\n❌ Sample Failures:")
        for f in failures[:10]:
            print(f" - {f}")

if __name__ == "__main__":
    main()
