
import numpy as np
from pathlib import Path
import mido
import ast
import src.chord_mapping as cm

# Copy of MODAL_SCALE_TYPES so it runs standalone
MODAL_SCALE_TYPES = {
    'type_0': {'name': 'Major', 'hc_distances': [0, 9, 9, 4, 9, 9, 9], 'description': 'Standard major scale (baseline 12-TET)'},
    'type_1': {'name': 'Neutral', 'hc_distances': [0, 8, 7, 7, 9, 8, 7], 'description': 'Neutral mode with neutral intervals'},
    'type_2': {'name': 'SubMinor', 'hc_distances': [0, 9, 3, 10, 9, 3, 10], 'description': 'Subminor mode'},
    'type_3': {'name': 'H_3rd_H_7th', 'hc_distances': [0, 9, 8, 5, 9, 8, 5], 'description': 'Harmonic 3rd and 7th mode'},
    'type_4': {'name': 'UpMajor', 'hc_distances': [0, 10, 9, 3, 9, 10, 9], 'description': 'Up-major mode'},
    'type_5': {'name': 'Major_v2', 'hc_distances': [0, 9, 9, 5, 8, 8, 9], 'description': 'Alternative major mode'},
    'type_6': {'name': 'Neutral_N', 'hc_distances': [0, 8, 7, 7, 9, 5, 10], 'description': 'Neutral N mode'}
}

def get_53tet_ratio(steps):
    return 2 ** (steps / 53.0)

def build_chromatic_scale_53tet(hc_distances, root_step=0, tonic_position=0):
    scale_7_steps = np.cumsum(hc_distances)
    relative_positions = [0, 2, 4, 5, 7, 9, 11]
    scale_positions = [(pos + tonic_position) % 12 for pos in relative_positions]
    chromatic_steps = [None] * 12
    for i, pos in enumerate(scale_positions):
        chromatic_steps[pos] = root_step + scale_7_steps[i]
    for i in range(12):
        if chromatic_steps[i] is None:
            prev_pos = -1
            next_pos = 12
            for j in range(i - 1, -1, -1):
                if chromatic_steps[j] is not None:
                    prev_pos = j
                    break
            for j in range(i + 1, 12):
                if chromatic_steps[j] is not None:
                    next_pos = j
                    break
            if prev_pos >= 0 and next_pos < 12:
                prev_step = chromatic_steps[prev_pos]
                next_step = chromatic_steps[next_pos]
                range_steps = next_step - prev_step
                positions = next_pos - prev_pos
                offset = i - prev_pos
                chromatic_steps[i] = round(prev_step + (range_steps * offset / positions))
            elif prev_pos >= 0:
                prev_step = chromatic_steps[prev_pos]
                range_steps = (root_step + 53) - prev_step
                positions = 12 - prev_pos
                offset = i - prev_pos
                chromatic_steps[i] = round(prev_step + (range_steps * offset / positions))
            else:
                chromatic_steps[i] = root_step + round((i / 12) * 53)
    return chromatic_steps

def calculate_53tet_frequency(midi_note, chromatic_scale_steps):
    tet12_freq = 440.0 * (2 ** ((midi_note - 69) / 12))
    note_in_octave = midi_note % 12
    step_53tet = chromatic_scale_steps[note_in_octave]
    step_12tet = (note_in_octave * 53) / 12
    hc_deviation = step_53tet - step_12tet
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

def map_chord_to_53tet(root, quality, chromatic_scale):
    key_to_position = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 
        'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 
        'B': 11, 'Cb': 11, 'E#': 5, 'Fb': 4, 'B#': 0
    }
    root_norm = root.replace('b', 'b') 
    if root_norm not in key_to_position:
        return f"{root} {quality}"
    root_pos = key_to_position.get(root_norm, 0)
    root_step_53 = chromatic_scale[root_pos]
    root_step_53 = root_step_53 % 53
    # Look up name
    root_name_53 = cm.NOTE_NAMES_53TET.get(root_step_53, f"Step{root_step_53}")
    
    if quality not in cm.CHORD_STRUCTURES_12TET:
        return f"{root_name_53} {quality}"
        
    intervals_12 = cm.CHORD_STRUCTURES_12TET[quality]
    chord_steps_abs = []
    
    for interval_semitones in intervals_12:
        note_pos = (root_pos + interval_semitones) % 12
        step_val = chromatic_scale[note_pos]
        # Octave logic
        expected_approx = interval_semitones * 4.4
        diff_direct = step_val - root_step_53
        diff_octave = (step_val + 53) - root_step_53
        if interval_semitones > 0:
             if abs(diff_octave - expected_approx) < abs(diff_direct - expected_approx):
                 step_val += 53
        chord_steps_abs.append(step_val)
        
    interval_sizes = [s - root_step_53 for s in chord_steps_abs]
    quality_map = {}
    for i, semitone in enumerate(intervals_12):
        quality_map[semitone] = interval_sizes[i]

    qualities = {}
    if 3 in quality_map: qualities['3rd'] = quality_map[3]
    elif 4 in quality_map: qualities['3rd'] = quality_map[4]
    
    if 7 in quality_map: qualities['5th'] = quality_map[7]
    elif 6 in quality_map: qualities['5th'] = quality_map[6]
    elif 8 in quality_map: qualities['5th'] = quality_map[8]
    
    if 10 in quality_map: qualities['7th'] = quality_map[10]
    elif 11 in quality_map: qualities['7th'] = quality_map[11]
    elif 9 in quality_map: qualities['7th'] = quality_map[9]
        
    q_names = []
    def get_q_name(steps):
        if steps <= 0: return "none"
        if steps in cm.INTERVAL_STEPS_TO_QUALITY: return cm.INTERVAL_STEPS_TO_QUALITY[steps]
        candidates = list(cm.INTERVAL_STEPS_TO_QUALITY.keys())
        if not candidates: return "unknown"
        closest = min(candidates, key=lambda x: abs(x - steps))
        return cm.INTERVAL_STEPS_TO_QUALITY[closest]

    q3_name = get_q_name(qualities.get('3rd', 0))
    if q3_name == "none": q3_name = "neutral" 
    q_names.append(q3_name)
    q5_name = get_q_name(qualities.get('5th', 0))
    if q5_name == "none": q5_name = "perfect"
    q_names.append(q5_name)
    if '7th' in qualities: q_names.append(get_q_name(qualities['7th']))
    
    lookup_tuple = tuple(q_names) 
    suffix = "???"
    if lookup_tuple in cm.CHORD_NAMING_TABLE:
        suffix = cm.CHORD_NAMING_TABLE[lookup_tuple]
    if suffix == "???": suffix = f"[{','.join(q_names)}]"
    return f"{root_name_53} {suffix}"

def process_text_file_conversion(midi_path, chromatic_scale, scale_type):
    input_path = Path(midi_path)
    text_filename = f"{input_path.stem}.txt"
    potential_dirs = [
        input_path.parent / "text_files",        
        input_path.parent.parent / "text_files", 
        input_path.parent.parent.parent / "text_files",
        Path("/Users/david/ANIMA_Data_Formation/dataset/text_files")
    ]
    text_path = None
    for d in potential_dirs:
         t = d / text_filename
         if t.exists():
             text_path = t
             break
             
    if text_path is None or not text_path.exists():
        print(f"ℹ️  No text file found for {input_path.name}")
        return

    print(f"📄 Processing text file: {text_path}")
    
    try:
        with open(text_path, 'r') as f:
            content = f.read()
            # Handle list-like strings safely
            try:
                data = ast.literal_eval(content)
            except:
                print("❌ Failed to parse text file.")
                return
        
        new_data = []
        i = 0
        while i < len(data):
            item = data[i]
            processed_chord = False
            if isinstance(item, str) and len(item) > 0 and item[0] in "ABCDEFG":
                if i + 1 < len(data):
                    next_item = data[i+1]
                    if next_item in cm.CHORD_STRUCTURES_12TET:
                        root = item
                        quality = next_item
                        new_name = map_chord_to_53tet(root, quality, chromatic_scale)
                        parts = new_name.split(' ', 1)
                        if len(parts) == 2:
                            new_data.append(parts[0]) 
                            new_data.append(parts[1]) 
                        else:
                            new_data.append(new_name)
                            new_data.append("")
                        i += 2
                        processed_chord = True
            if not processed_chord:
                new_data.append(item)
                i += 1
                
        output_filename = f"{input_path.stem}_{scale_type}.txt"
        output_path = text_path.parent / output_filename
        with open(output_path, 'w') as f:
            f.write(str(new_data))
        print(f"✅ Text conversion saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing text file: {e}")

def convert_midi_to_53tet(input_midi_path, scale_type='type_1', output_dir=None, key=None):
    print(f"DEBUG: Running debug conversion for {scale_type}")
    input_path = Path(input_midi_path)
    if key is None:
        import re
        match = re.search(r'_([A-G][#b]?)_(?:major|minor)', input_path.stem)
        key = match.group(1) if match else 'C'
    
    tonic_position = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}.get(key[0], 0)
    config = MODAL_SCALE_TYPES[scale_type]
    chromatic_scale = build_chromatic_scale_53tet(config['hc_distances'], root_step=0, tonic_position=tonic_position)
    
    process_text_file_conversion(input_path, chromatic_scale, scale_type)
    
    # Skip MIDI generation for speed/simplicity in this debug run
    # (The user wanted the text file mainly)
    return

if __name__ == "__main__":
    input_midi = "/Users/david/ANIMA_Data_Formation/dataset/midi_files/mpe/47832_Something_C_major.mid"
    convert_midi_to_53tet(input_midi, scale_type="type_4")
