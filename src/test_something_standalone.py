
import sys
import importlib
import numpy as np
import os

# Add src to path
sys.path.append(os.getcwd())

import voicing
import xmlTranslator as xmlT
from utils import get_project_root

def run_test():
    # Use the same path logic as notebook
    directory = '/dataset/iRealXML'
    # get_project_root returns pathlib Path
    root = get_project_root()
    # Ensure root is string if needed or concat properly
    myPath = str(root) + directory

    print(f"Loading dataset from {myPath}...")
    # This might take a moment
    chords, durations, meta = xmlT.parse_info_from_XML(myPath)

    song_id = next((i for i, m in enumerate(meta) if m['song_name'] == 'Something'), None)

    if song_id is None:
        print("Something not found!")
        return

    print(f"Found 'Something' at index {song_id}")
    song = chords[song_id]
    
    # Normalized replacements
    song = xmlT.replaceTheseChords([song], False)[0]

    # Fix compound tokens logic from notebook
    def fix_compound_tokens(token_list):
        fixed = []
        for token in token_list:
            if isinstance(token, str) and ' add ' in token:
                parts = token.split(' add ')
                fixed.append(parts[0])
                fixed.append('add ' + parts[1])
            elif isinstance(token, str) and ' alter ' in token:
                parts = token.split(' alter ')
                fixed.append(parts[0])
                fixed.append('alter ' + parts[1])
            else:
                fixed.append(token)
        return fixed

    song = fix_compound_tokens(song)
    print(f"Processing {len(song)} tokens...")

    # Process with Voicing
    print("Running Voicing...")
    v = voicing.Voicing()

    # Add 'maj' tokens (logic from notebook)
    processed = []
    for i, token in enumerate(song):
        processed.append(token)
        if i < len(song) - 1:
            next_t = song[i+1]
            prev_t = song[i-1] if i > 0 else ''
            
            # Helper to check if structural
            is_struct = (next_t in v.structural_elements or str(next_t).startswith('Form_'))
            
            if token in v.all_notes and next_t != 'N.C.' and prev_t != '/' and is_struct:
                processed.append('maj')

    midi_seq, status = v.convert_chords_to_voicing(processed)
    print(f"Generated {len(midi_seq)} chords.")
    
    # Check if we have notes
    notes_count = sum(1 for m,d,l in midi_seq if len([n for n in m if n>0]) > 0)
    print(f"Chords with notes: {notes_count}")

    # Export
    out_name = "VALIDATION_Something_Standalone"
    out_dir = "../dataset/midi_files/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    v.export_to_midi(midi_seq, out_name, out_dir)
    print(f"Exported to {out_dir}{out_name}.mid")

if __name__ == "__main__":
    run_test()
