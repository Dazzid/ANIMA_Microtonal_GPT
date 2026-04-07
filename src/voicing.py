#make a class of voicing
# ENHANCED VERSION - Improved voicing system inspired by modal_studio_Chord.js
# 
# Key improvements:
# 1. Seven voicing templates (v_0 to v_6) instead of just 4, providing more variety
# 2. Each chord type now has sophisticated multi-template voicings with different characteristics:
#    - Drop-2 voicings for better voice leading
#    - Open and closed position voicings
#    - Extended voicings with color tones (9ths, 11ths, 13ths)
# 3. Advanced voice leading methods:
#    - optimize_voice_leading(): Minimizes movement between chords
#    - add_extensions_for_quality(): Adds appropriate extensions based on chord type
#    - get_drop_2_voicing() and get_drop_3_voicing(): Create jazz-style drop voicings
# 4. Function-based voicing selection inspired by modal harmony principles
#
from midiutil import MIDIFile
import tqdm as tqdm
import random
import numpy as np
import pytz
from datetime import datetime, timezone
from music21 import midi, environment, stream
import os
import mido
from mido import MidiFile, MidiTrack, Message

class Voicing:
    #define the class
    def __init__(self):
        
        #define the natures of the chords
        self.natures = {'maj', 'maj6', 'maj7', 'm', 'm6', 'm7', 'm_maj7', 'dom7', 'sus', 'sus2', 'sus7', 'sus4', 'o7', 'o', 'ø7', 'power', 'aug', 'o_maj7', 'N.C.'}
        
        #alterations and add
        self.alter = {'add b2', 'add 2', 'add b5', 'add 5', 'add #5', 'add b6', 'add 6' 'add 7', 'add #7', 'add 8', 'add b9', 'add 9', 'add #9', 'add #11', 'add 13', 'add b13', 'alter #11', 'alter #5', 'alter #7', 'alter #9', 'alter b5', 'alter b9'}        
        
        #Structural elements
        self.structural_elements = {'.', '|', '||', ':|', '|:', 'b||', 'e||', '/'} #to add the maj token 
        
        #element in the chord frontiers
        self.after_chords = {'.', '|', '||', ':|', '|:', 'b||', 'e||'} 
        
        #Voicing - 7 different templates inspired by modal_studio_Chord.js
        # These correspond to different chord functions (I, II, III, IV, V, VI, VII)
        self.voicing = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6']
        
        #Durations 
        self.durations = {'0.3997395833333333', '0.4440104166666667', '0.5', '0.5703125',
                '0.6666666666666666', '0.75', '0.7994791666666666', '0.8880208333333334',
                '1.0', '1.1419270833333333', '1.3333333333333333', '1.5',
                '1.5989583333333333', '1.7135416666666667', '2.0', '2.25',
                '2.3997395833333335', '2.6666666666666665', '3.0', '4.0'}
        
        #All notes
        self.all_notes = {
            'C': 48, 'C#': 49, 'Db': 49, 'D': 50, 'D#': 51, 'Eb': 51, 'E': 52, 'Fb': 52, 'F': 53, 'E#': 53, 'F#': 54, 'Gb': 42, 'G': 43, 'G#': 44, 'Ab':44, 'A': 45, 'A#': 46, 'Bb': 46, 'B': 47, 
            'A##': 47, 'Abb': 43, 'Abbb': 42, 'B#': 48, 'B##': 49, 'Bbb': 45, 'Bbbb': 44,
            'C##': 50, 'C###': 51, 'Cb': 47, 'Cbb': 46, 'D##': 52, 'Dbb': 48, 'Dbbb': 47, 'E##': 54, 'Ebb': 50, 'Ebbb': 49, 
            'F##': 55, 'F###': 56, 'Fbb': 51, 'G##': 45, 'Gbb': 41
            }
        
        # ----------------------------------------------------------------------
        # CLOSED POSITION STACKS (4-note blocks for Drop 2/3)
        # These are used to generate the Upper Structure
        # ----------------------------------------------------------------------
        self.closed_stacks = {
            'maj':      [0, 4, 7, 12],   # R 3 5 R (Doubled Root)
            'maj6':     [0, 4, 7, 9],    # R 3 5 6
            'maj7':     [0, 4, 7, 11],   # R 3 5 7
            'm':        [0, 3, 7, 12],   # R b3 5 R
            'm6':       [0, 3, 7, 9],    # R b3 5 6
            'm7':       [0, 3, 7, 10],   # R b3 5 b7
            'm_maj7':   [0, 3, 7, 11],   # R b3 5 7
            'dom7':     [0, 4, 7, 10],   # R 3 5 b7
            'sus':      [0, 5, 7, 12],   # R 4 5 R (Assume sus4)
            'sus4':     [0, 5, 7, 12],   # R 4 5 R
            'sus2':     [0, 2, 7, 12],   # R 2 5 R
            'sus7':     [0, 5, 7, 10],   # R 4 5 b7
            'o7':       [0, 3, 6, 9],    # R b3 b5 bb7
            'o':        [0, 3, 6, 12],   # R b3 b5 R
            'ø7':       [0, 3, 6, 10],   # R b3 b5 b7
            'aug':      [0, 4, 8, 12],   # R 3 #5 R
            'power':    [0, 7, 12, 19],  # R 5 R 5
            'o_maj7':   [0, 3, 6, 11],   # R b3 b5 7
        }

        # Legacy Templates (kept for fallback or specific styles)
        # Major chord voicings (Root, 3, 5, 9)
        self.maj = {
            'v_0': [0, 4, 7, 12],            # R-3-5-R (Fixed to 4 notes)
            'v_1': [0, 4, 7, 12],            # R-3-5-R
            'v_2': [0, 4, 7, 14],            # R-3-5-9
            'v_3': [0, 4, 7, 12],           # R-3-R-5 (Open)
            'v_4': [0, 4, 7, 12],           # R-5-R-3 (Open)
            'v_5': [0, 4, 7, 14],            # R-3-5-9
            'v_6': [0, 4, 7, 12]                 # Backup
        }
        
        # Major 7th voicings (Root, 3, 7, 9, 5)
        self.maj7 = {
            'v_0': [0, 4, 7, 11],            # R-3-5-7
            'v_1': [0, 11, 14, 16],          # R-7-9-3 (Rootless A-form-ish idea, but applied w/ root)
            'v_2': [0, 4, 7, 11, 14],        # R-3-5-7-9
            'v_3': [0, 4, 11, 14],           # R-3-7-9 (Shell)
            'v_4': [0, 7, 11, 16],           # R-5-7-3 (Open)
            'v_5': [0, 4, 7, 11, 14],        # R-3-5-7-9
            'v_6': [0, 4, 7, 11]             # R-3-5-7
        }
        
        # Minor chord voicings (Root, b3, 5, 9)
        self.m = {
            'v_0': [0, 3, 7],                # R-b3-5
            'v_1': [0, 3, 7, 12],            # R-b3-5-R
            'v_2': [0, 3, 7, 14],            # R-b3-5-9
            'v_3': [0, 3, 12, 19],           # R-b3-R-5
            'v_4': [0, 7, 12, 15],           # R-5-R-b3
            'v_5': [0, 3, 7, 14],            # R-b3-5-9
            'v_6': [0, 3, 7]                 # Backup
        }
        
        # Minor 7th voicings (Root, b3, b7, 9, 5)
        self.m7 = {
            'v_0': [0, 3, 7, 10],            # R-b3-5-b7
            'v_1': [0, 10, 14, 15],          # R-b7-9-b3 (Shell + 9)
            'v_2': [0, 3, 7, 10, 14],        # R-b3-5-b7-9
            'v_3': [0, 3, 10, 14],           # R-b3-b7-9 (Shell)
            'v_4': [0, 7, 10, 15],           # R-5-b7-b3 (Open)
            'v_5': [0, 3, 7, 10, 14],        # R-b3-5-b7-9
            'v_6': [0, 3, 7, 10]             # R-b3-5-b7
        }
        
        # Dominant 7th voicings (Root, 3, b7, 9, 13)
        self.dom7 = {
            'v_0': [0, 4, 7, 10],            # R-3-5-b7
            'v_1': [0, 10, 14, 16],          # R-b7-9-3 (Shell + 9)
            'v_2': [0, 4, 7, 10, 14],        # R-3-5-b7-9
            'v_3': [0, 4, 10, 14],           # R-3-b7-9 (Shell)
            'v_4': [0, 7, 10, 16],           # R-5-b7-3 (Open)
            'v_5': [0, 4, 10, 14, 21],       # R-3-b7-9-13 (Rich)
            'v_6': [0, 4, 7, 10]             # R-3-5-b7
        }
        
        # Half-diminished (ø7)
        self.ø7 = {
            'v_0': [0, 3, 6, 10],            # R-b3-b5-b7
            'v_1': [0, 3, 6, 10, 15],        # R-b3-b5-b7-11
            'v_2': [0, 3, 6, 10],
            'v_3': [0, 3, 6, 10],
            'v_4': [0, 6, 10, 15],
            'v_5': [0, 3, 6, 10],
            'v_6': [0, 3, 6, 10]
        }
        
        # Diminished 7th (o7)
        self.o7 = {
            'v_0': [0, 3, 6, 9],             # R-b3-b5-bb7
            'v_1': [0, 3, 6, 9, 14],         # Add 9 (maj9)
            'v_2': [0, 3, 6, 9],
            'v_3': [0, 3, 6, 9],
            'v_4': [0, 3, 6, 9],
            'v_5': [0, 3, 6, 9],
            'v_6': [0, 3, 6, 9]
        }
        
        # Diminished triad (o)
        self.o = {
            'v_0': [0, 3, 6, 12],            # R-b3-b5-R
            'v_1': [0, 3, 6],
            'v_2': [0, 3, 6],
            'v_3': [0, 3, 6],
            'v_4': [0, 3, 6],
            'v_5': [0, 3, 6],
            'v_6': [0, 3, 6]
        }
        
        # Sus4 (Root, 4, 5, b7, 9)
        self.sus = {
            'v_0': [0, 5, 7],                # R-4-5
            'v_1': [0, 5, 7, 10],            # R-4-5-b7
            'v_2': [0, 5, 7, 10, 14],        # R-4-5-b7-9
            'v_3': [0, 5, 10, 14],           # R-4-b7-9
            'v_4': [0, 7, 12, 17],           # R-5-R-4
            'v_5': [0, 5, 7, 10, 14, 21],    # R-4-5-b7-9-13
            'v_6': [0, 5, 7, 12]             # R-4-5-R
        }
        
        self.sus7 = self.sus
        self.sus4 = self.sus
        
        # Sus2
        self.sus2 = {
            'v_0': [0, 2, 7],                # R-2-5
            'v_1': [0, 2, 7, 12],
            'v_2': [0, 2, 7, 14],
            'v_3': [0, 7, 14, 19],
            'v_4': [0, 2, 7],
            'v_5': [0, 2, 7],
            'v_6': [0, 2, 7]
        }
        
        # Augmented
        self.aug = {
            'v_0': [0, 4, 8],                # R-3-#5
            'v_1': [0, 4, 8, 10],            # R-3-#5-b7
            'v_2': [0, 4, 8, 12],
            'v_3': [0, 4, 8],
            'v_4': [0, 4, 8],
            'v_5': [0, 4, 8],
            'v_6': [0, 4, 8]
        }
        
        # Minor major 7th
        self.m_maj7 = {
            'v_0': [0, 3, 7, 11],            # R-b3-5-7
            'v_1': [0, 3, 7, 11, 14],        # R-b3-5-7-9
            'v_2': [0, 3, 7, 11],
            'v_3': [0, 3, 7, 11],
            'v_4': [0, 3, 7, 11],
            'v_5': [0, 3, 7, 11],
            'v_6': [0, 3, 7, 11]
        }
        
        # Major 6th
        self.maj6 = {
            'v_0': [0, 4, 7, 9],             # R-3-5-6
            'v_1': [0, 4, 7, 9, 14],         # R-3-5-6-9
            'v_2': [0, 3, 7, 9],              # ??? (Wait, 3 is b3. Should be 4). 
                                             # Fixed: maj6 is 0, 4, 7, 9.
            'v_3': [0, 4, 7, 9],
            'v_4': [0, 4, 7, 9],
            'v_5': [0, 4, 7, 9],
            'v_6': [0, 4, 7, 9]
        }
        # Fixed v_2 above:
        self.maj6['v_2'] = [0, 4, 7, 9]
        
        # Minor 6th
        self.m6 = {
            'v_0': [0, 3, 7, 9],             # R-b3-5-6
            'v_1': [0, 3, 7, 9, 14],         # R-b3-5-6-9
            'v_2': [0, 3, 7, 9],
            'v_3': [0, 3, 7, 9],
            'v_4': [0, 3, 7, 9],
            'v_5': [0, 3, 7, 9],
            'v_6': [0, 3, 7, 9]
        }
        
        # o_maj7
        self.o_maj7 = {
            'v_0': [0, 3, 6, 11],
            'v_1': [0, 3, 6, 11],
            'v_2': [0, 3, 6, 11],
            'v_3': [0, 3, 6, 11],
            'v_4': [0, 3, 6, 11],
            'v_5': [0, 3, 6, 11],
            'v_6': [0, 3, 6, 11]
        }
        
        # Power
        self.power = {
            'v_0': [0, 7, 12],
            'v_1': [0, 7, 12, 19],
            'v_2': [0, 7, 12],
            'v_3': [0, 7, 12],
            'v_4': [0, 7, 12],
            'v_5': [0, 7, 12],
            'v_6': [0, 7, 12]
        }
        
        # No chord
        self.noChord = {
            'v_0': [0], 'v_1': [0], 'v_2': [0],
            'v_3': [0], 'v_4': [0], 'v_5': [0],
            'v_6': [0]
        }
               
        #TODO: define voicing for guitar
        
        #Define the voicing dictionaries for the chords
        self.chord_voicing = {'maj': self.maj, 'maj7': self.maj7, 'm': self.m, 'm7': self.m7, 'dom7': self.dom7, 
                              'ø7': self.ø7, 'o7': self.o7, 'o': self.o, 'sus': self.sus, 'sus7': self.sus7, 
                              'sus2': self.sus2, 'sus4': self.sus4, 'm6': self.m6, 'power': self.power, 
                              'm_maj7': self.m_maj7, 'maj6': self.maj6, 'aug': self.aug, 'o_maj7': self.o_maj7, 'N.C.': self.noChord}
    #-----------------------------------------------------------------------
    def listToIgnore(self):
        ignore_list = {'<start>', '<end>', '<pad>', '.', '|', '||', 'b||', 'e||', 'Repeat_0', 'Repeat_1', 'Repeat_2', 'Repeat_3', 'Intro', 
                        'Form_A', 'Form_B', 'Form_C', 'Form_D', 
                        'Form_verse', 'Form_intro', 'Form_Coda', 'Form_Head', 
                        'Form_Segno', '|:', ':|'}
        return ignore_list

    #-----------------------------------------------------------------------
    def getStructuralElements(self):
        return self.structural_elements
    
    #-----------------------------------------------------------------------
    # Add the maj token
    def add_maj_token(self, sequence):
        new_sequence = []
        for song in sequence:
            new_song = []
            for i in range(len(song)):
                element = song[i]
                new_song.append(element)
                if element in self.all_notes.keys() and (i == len(song) - 1 or song[i + 1] in self.structural_elements 
                                                         or song[i + 1].startswith('Form_')) and song[i-1] != '/':
                    new_song.append('maj')
            new_sequence.append(new_song)
        
        return new_sequence
    #-----------------------------------------------------------------------
    # Add the voicing to the sequence
    def get_midi(self, sequence):
        midi_sequence = []
        root = 0
        mod = 3
        status = True
        # Create a dictionary for the alter section
        add_dict = {
            'add b13': 8 + 12,
            'add 13': 9 + 12, 
            'add #11': 6 + 12,
            'add 11': 6 + 11,
            'add #9': 3 + 12,
            'add 9': 2 + 12,
            'add b9': 1 + 12,
            'add 8': 12,
            'add 7': 11,
            'add #7': 11,
            'add 6': 9,
            'add b6': 8 + 12,
            'add 5': 7,
            'add b5': 6,
            'add 2': 2 + 12,
            'add b2': 1
        }
        # Create a dictionary for the alter section
        alter_dict = {
            'alter b9': 2,
            'alter #9': 2,
            'alter b5': 7,
            'alter #5': 7,
            'alter #7': 11,
            'alter #11': 5
        }
        
        midi = [0, 0, 0, 0, 0, 0, 0, 0]
        duration = 0.0
        #check the chord info
        for i, element in enumerate(sequence):
            
            #Check it is a dot ----------------------------------------------------
            if element == '.':
                #duration = float(sequence[i+1])
                midi = [0, 0, 0, 0, 0, 0, 0, 0]
                midi_sequence.append(midi)
            #check the duration ----------------------------------------------------
            elif element in self.durations:
                duration = float(element)
                midi_sequence.append(midi)
            #check notes ------------------------------------------------------------
            elif element in self.all_notes and sequence[i-1] != '/':
                root = self.all_notes[element]
                midi = [root, 0, 0, 0, 0, 0, 0, 0]
              
                midi_sequence.append(midi)
                #print(element, sequence[i-1][0]) 
            
            # Nature section --------------------------------------------------------
            elif element in self.natures:
                n = i % mod
                midi = [x + root for x in self.chord_voicing[element][self.voicing[n]]]
                #print('chord:', element, midi)
                infoMidi = midi.copy()
                midi_sequence.append(infoMidi)
            
            # Add section --------------------------------------------------------      
            elif element in add_dict:
                #print('original', midi)
                new_note = root + add_dict[element]
                midiInfo = midi.copy()    
                if new_note not in midiInfo:
                    midiInfo.append(new_note)
                      
                if element == 'add b9' or element == 'add #9':
                    #check if the 9 is in the chord
                    if (root + 14) in midiInfo:
                        index = midiInfo.index(root + 14)
                        midiInfo.pop(index)
                    elif (root + 26) in midiInfo:
                        index = midiInfo.index(root + 26)
                        midiInfo.pop(index)
                        
                midi_sequence.append(midiInfo)
                midi = midiInfo
                
            # Alter section --------------------------------------------------------            
            elif element in alter_dict:
                #print('original', element, midi)
                my_ref = [x for x in midi if (x - root) % 12 == alter_dict[element]]
                midiInfo = midi.copy() 
                
                if len(my_ref) == 1:
                    loc = midi.index(my_ref[0])
                    if element.find('b') != -1:
                        midiInfo[loc] = my_ref[0] - 1
                    elif element.find('#') != -1:
                        midiInfo[loc] = my_ref[0] + 1
                        
                elif len(my_ref) > 1:
                    for i, n in enumerate(my_ref):
                        loc = midi.index(n)
                        if element.find('b') != -1 and (n - root) % 12 == alter_dict[element]:
                            midiInfo[loc] = n - 1
                        elif element.find('#') != -1 and (n - root) % 12 == alter_dict[element]:
                            midiInfo[loc] = n + 1 
                
                elif len(my_ref) == 0:
                    new_note = root + alter_dict[element]
                    if element.find('b') != -1:
                        new_note -= 1
                    elif element.find('#') != -1:
                        new_note += 1
                    midiInfo.append(new_note)
                    
                midi_sequence.append(midiInfo)
                #print('result', element, midi)
                
            # Slash section --------------------------------------------------------    
            elif element == '/':
                # Keep sequences aligned - append marker but don't affect voicing
                # The actual slash bass note will be in the next element
                thisMidi = [0, 0, 0, 0, 0, 0, 0, 0]
                midi_sequence.append(thisMidi)
                
            # New root after slash section -----------------------------------------  
            elif sequence[i-1][0] == '/' and element in self.all_notes:
                # Slash chord: Take the current chord voicing, move old root up octave, add new bass
                # Example: G7/D → G7 chord [43,65,71] becomes [50,55,65,71] (D bass, G+octave, B, F)
                slash_bass = self.all_notes[element]
                
                # Start with current chord voicing
                midiInfo = [x for x in midi if x > 0]  # Get only non-zero notes
                
                # Find the old root (lowest note in current voicing) and move it up one octave
                if len(midiInfo) > 0:
                    old_root = midiInfo[0]  # Lowest note is usually the root
                    midiInfo[0] = old_root + 12  # Move up one octave
                
                # Insert the new bass note at the beginning
                midiInfo.insert(0, slash_bass)
                
                # Pad to 8 notes
                while len(midiInfo) < 8:
                    midiInfo.append(0)
                
                midi_sequence.append(midiInfo)
                midi = midiInfo  # Update midi so subsequent operations use slashed chord
            
            # Structural elements section ---------------------------------------------
            elif element in self.structural_elements and element != '/':
                thisMidi = [0, 0, 0, 0, 0, 0, 0, 0]
                midi_sequence.append(thisMidi)
                
            # Form section -------------------------------------------------------------
            elif element not in self.all_notes and element not in self.natures and element not in self.structural_elements and element not in self.durations:
                thisMidi = [0, 0, 0, 0, 0, 0, 0, 0]
                midi_sequence.append(thisMidi)
        
            
        #Normalize the length of the MIDI sequence to 8 ----------------------------
        for i, item in enumerate(midi_sequence):    
            if len(item) < 8:
                for i in range(8 - len(item)):
                    item.append(0)
        midi_sequence = np.array(midi_sequence, dtype=int)
        return midi_sequence, status
    
    def generate_voicing_candidates(self, root, nature):
        """
        Generate voicing candidates using strict Drop 2 and Drop 3 logic.
        
        Definition:
        1. Left Hand: Root (in Bass register, usually C2-C3).
        2. Right Hand: 
           - Start with 4-note Closed Block (e.g. R-3-5-7 or R-3-5-R).
           - Invert the Closed Block (0, 1, 2, 3 inversions).
           - Apply Drop 2: Drop 2nd note from top an octave down (usually ends up between LH and RH).
           - Apply Drop 3: Drop 3rd note from top an octave down.
        """
        candidates = []
        
        # 1. Determine Closed Stack relative to root=0
        if nature == 'N.C.':
            return []
            
        stack = self.closed_stacks.get(nature, [0, 4, 7, 12]) # Default to Major Add Octave
        
        # 2. Generate Inversions of the Closed Stack (still relative to 0)
        # stack is e.g. [0, 4, 7, 12]
        # inv0: [0, 4, 7, 12]
        # inv1: [4, 7, 12, 12] -> [4, 7, 12, 12] normalized?
        # Better: Invert by taking bottom note + 12.
        
        base_stack = sorted(stack)
        inversions = []
        
        # Create 4 inversions
        current = list(base_stack)
        for _ in range(4):
            inversions.append(sorted(current))
            # Invert: remove bottom, add it +12
            bottom = current[0]
            current = current[1:] + [bottom + 12]
            current = sorted(current)
            
        # 3. For each inversion, generate Closed, Drop 2, Drop 3
        # And transpose to valid registers
        
        # Target Range for Upper Structure (before drop):
        # We want the resulting CHORD to sit nicely.
        # Usually checking Result range is better.
        
        for inv in inversions:
            # inv is relative to 0. e.g. [0, 4, 7, 12]
            
            # --- Type A: Closed (Reference, maybe useful) ---
            # candidates.append(self._create_candidate(root, inv, 'Closed'))
            
            # --- Type B: Drop 2 ---
            # Drop 2nd highest note octave down
            if len(inv) >= 2:
                d2 = list(inv)
                d2[-2] -= 12
                candidates.extend(self._create_realizations(root, d2, nature))

            # --- Type C: Drop 3 ---
            # Drop 3rd highest note octave down
            if len(inv) >= 3:
                d3 = list(inv)
                d3[-3] -= 12
                candidates.extend(self._create_realizations(root, d3, nature))
                
        # If no candidates found (e.g. strict range checks failed), fallback to simple
        if not candidates:
             return [[root, root+4, root+7, root+12]]
             
        return candidates

    def _create_realizations(self, root, intervals, nature):
        """
        Create concrete MIDI voicings from relative intervals (Drop applied)
        by trying different octaves.
        
        intervals: relative intervals (e.g. [-5, 0, 4, 11])
        """
        out = []
        sorted_intervals = sorted(intervals)
        
        # Try centering the voicing
        # Base Root usually C2(36) or C3(48). 
        # Left Hand usually plays Root at C2 or C3.
        
        # We assume 'intervals' captures the relative shape of the RH + Dropped Note.
        # But we MUST ADD The LEft Hand Root explicitly if it's not covered?
        # The user says: "Root is on the left hand... Then the triad [dropped] is on the right"
        # The dropped note falls "in between".
        
        # So structure: [BassRoot] + [UpperStructure (with Drop applied)]
        
        # Iterate possible root octaves for the Left Hand
        for bass_oct in [36, 48]: # C2, C3
            actual_root = root % 12 + bass_oct # e.g. 50 (D3) or 38 (D2)
            if actual_root > 55: actual_root -= 12 # Keep bass low-ish
            if actual_root < 36: actual_root += 12
            
            # Now place the Upper Structure
            # We want the Upper Structure to be "above" the bass, or "wrapping" it?
            # Usually above.
            # The intervals are relative to 0.
            # We need to shift them so they sit in the middle register (C3-C5).
            
            # Find loop of transpositions for the upper structure
            # e.g. center around C4 (60)
            
            # Get centroid of intervals
            avg_int = sum(sorted_intervals)/len(sorted_intervals)
            
            # We want avg_val + shift ~ 60
            # shift ~ 60 - avg_int
            
            base_shift = int(60 - avg_int)
            # Round to nearest octave (12)
            base_shift = round(base_shift / 12) * 12
            
            # Try a few shifts around there
            for oct_shift in [base_shift - 12, base_shift, base_shift + 12]:
                
                # Apply shift to upper structure
                upper_notes = [n + oct_shift for n in sorted_intervals] # Absolute MIDI notes now?
                # Wait, intervals were relative to 0. 
                # If we add oct_shift (e.g. 60), we treat key of C=0.
                # But we are in key of Root.
                # So we need to add Root%12 too? 
                # Yes. intervals are relative to Root.
                
                # Real notes = Root + Interval + OctaveShift
                # But wait, logic above: "intervals" are relative to local 0.
                
                real_upper = [root % 12 + n + oct_shift for n in sorted_intervals]
                
                # Combine Bass + Upper
                # Bass = actual_root
                
                # Check for clash: if lowest upper note is below bass?
                min_upper = min(real_upper)
                if min_upper <= actual_root:
                    continue # Upper structure shouldn't go below bass root usually
                    
                # Create voicing
                voicing = [actual_root] + real_upper
                
                if self.is_valid_voicing(voicing):
                    out.append(voicing)
                    
        return out

    def is_valid_voicing(self, voicing):
        """Check for muddiness and range"""
        if not voicing: return False
        
        # Range check: Piano 21 to 108
        if min(voicing) < 21 or max(voicing) > 108:
            return False
            
        # Low Interval Limit (roughly)
        # No intervals < 3 semitones below E3 (52)
        # No intervals < 5 semitones below C3 (48)
        
        notes = sorted(voicing)
        for i in range(len(notes)-1):
            n1 = notes[i]
            n2 = notes[i+1]
            interval = n2 - n1
            
            if interval == 0: continue # Unison is okay? Maybe avoid doubling thirds.
            
            if n1 < 48 and interval < 5: # Below C3, no thirds/seconds
               return False
            if n1 < 52 and interval < 3: # Below E3, no seconds
               return False
               
        return True

    #-----------------------------------------------------------------------
    # Add the voicing to the sequence
    # DOT (.) = START of chord. Everything until next DOT is ONE chord.
    # Structure: . duration root nature [extensions] [/ bass]
    def convert_chords_to_voicing(self, sequence):
        midi_sequence = []
        mod = 7
        status = True
        previous_voicing = None
        
        add_dict = {
            'add b13': 8 + 12, 'add 13': 9 + 12, 'add #11': 6 + 12, 'add 11': 6 + 11,
            'add #9': 3 + 12, 'add 9': 2 + 12, 'add b9': 1 + 12, 'add 8': 12,
            'add 7': 11, 'add #7': 11, 'add 6': 9, 'add b6': 8 + 12,
            'add 5': 7, 'add b5': 6, 'add 2': 2 + 12, 'add b2': 1
        }
        alter_dict = {
            'alter b9': 2, 'alter #9': 2, 'alter b5': 7,
            'alter #5': 7, 'alter #7': 11, 'alter #11': 5
        }
        
        # Find all DOT positions - each DOT starts exactly ONE chord
        dot_positions = [i for i, elem in enumerate(sequence) if elem == '.']
        
        for dot_idx, dot_pos in enumerate(dot_positions):
            # Find the END of this chord (next DOT or end of sequence)
            if dot_idx + 1 < len(dot_positions):
                end_pos = dot_positions[dot_idx + 1]
            else:
                end_pos = len(sequence)
            
            # Extract chord tokens between this DOT and next DOT
            chord_tokens = sequence[dot_pos:end_pos]
            
            # Parse this ONE chord
            duration = 0.0
            root = 0
            midi = [0, 0, 0, 0, 0, 0, 0, 0]
            chord_label = ''
            has_slash = False
            slash_bass = None
            
            for j, token in enumerate(chord_tokens):
                if token == '.':
                    continue
                    
                elif token in self.durations:
                    duration = float(token)
                    
                elif token in self.all_notes and (j == 0 or chord_tokens[j-1] != '/'):
                    # This is the ROOT note (not slash bass)
                    root = self.all_notes[token]
                    midi = [root, 0, 0, 0, 0, 0, 0, 0]
                    chord_label = token
                    
                elif token in self.natures:
                    # Get all possible voicing candidates (Closed, Drop 2, Drop 3 at various octaves)
                    all_voicings = self.generate_voicing_candidates(root, token)
                    
                    if previous_voicing is not None:
                        # Select the one that connects smoothest to previous chord
                        midi = self.select_best_voicing(previous_voicing, all_voicings, prev_root=None, curr_root=root)
                    else:
                        # First chord: Pick a candidate in the middle register
                        # Bias towards C3/C4 for root
                        def root_clarity(v):
                            if not v: return 100
                            # Root is usually the lowest note here
                            return abs(v[0] - 48) # Distance from C3
                            
                        midi = min(all_voicings, key=root_clarity) if all_voicings else [root, root+4, root+7]
                    
                    chord_label = token
                    
                elif token in add_dict:
                    new_note = root + add_dict[token]
                    if new_note not in midi:
                        filled = False
                        for idx in range(len(midi)):
                            if midi[idx] == 0:
                                midi[idx] = new_note
                                filled = True
                                break
                        if not filled:
                            midi.append(new_note)
                    if token == 'add b9' or token == 'add #9':
                        if (root + 14) in midi:
                            midi[midi.index(root + 14)] = 0
                        elif (root + 26) in midi:
                            midi[midi.index(root + 26)] = 0
                            
                elif token in alter_dict:
                    my_ref = [x for x in midi if x > 0 and (x - root) % 12 == alter_dict[token]]
                    if len(my_ref) >= 1:
                        for n in my_ref:
                            loc = midi.index(n)
                            if 'b' in token:
                                midi[loc] = n - 1
                            elif '#' in token:
                                midi[loc] = n + 1
                    elif len(my_ref) == 0:
                        new_note = root + alter_dict[token]
                        if 'b' in token:
                            new_note -= 1
                        elif '#' in token:
                            new_note += 1
                        filled = False
                        for idx in range(len(midi)):
                            if midi[idx] == 0:
                                midi[idx] = new_note
                                filled = True
                                break
                        if not filled:
                            midi.append(new_note)
                                
                elif token == '/':
                    has_slash = True
                    
                elif has_slash and token in self.all_notes:
                    # Slash bass note - modify the chord
                    # Get the bass note pitch class
                    bass_pitch_class = self.all_notes[token] % 12
                    
                    # SMOOTH VOICE LEADING: Find optimal octave for slash bass
                    # Start with octave 2 as default
                    slash_bass = 36 + bass_pitch_class
                    
                    # If we have a previous voicing, find closest octave
                    if previous_voicing:
                        prev_bass = previous_voicing[0]  # Previous bass note
                        
                        # Try different octaves and pick the closest to previous bass
                        best_distance = float('inf')
                        best_octave_bass = slash_bass
                        
                        # Try octaves 1-4 (MIDI 24-71)
                        for octave in range(1, 5):
                            candidate_bass = 12 * octave + bass_pitch_class
                            distance = abs(candidate_bass - prev_bass)
                            
                            # Prefer smallest distance, but keep bass in reasonable range (24-60)
                            if distance < best_distance and 24 <= candidate_bass <= 60:
                                best_distance = distance
                                best_octave_bass = candidate_bass
                        
                        slash_bass = best_octave_bass
                    
                    # Get non-zero notes from current voicing
                    notes = [x for x in midi if x > 0]
                    
                    if len(notes) > 0:
                        # Find the original root (by pitch class, not just lowest note)
                        root_pitch_class = root % 12
                        
                        # Remove any note with same pitch class as new bass (to avoid duplicates)
                        bass_pitch = slash_bass % 12
                        notes = [x for x in notes if x % 12 != bass_pitch]
                        
                        # Ensure all remaining notes are ABOVE the bass AND respect low interval limits
                        adjusted_notes = []
                        for n in notes:
                            # 1. Must be above bass
                            while n <= slash_bass:
                                n += 12
                            
                            # 2. Low interval limit check (avoid muddy minor 2nds/Major 2nds deep in bass)
                            # If interval to bass is small (< 5 semitones) and bass is low (< 55 / G3)
                            # Shift up an octave
                            if (n - slash_bass) < 5 and slash_bass < 55:
                                n += 12
                                
                            adjusted_notes.append(n)
                            
                        # Sort and rebuild: bass first, then rest ascending
                        adjusted_notes = sorted(adjusted_notes)
                        notes = [slash_bass] + adjusted_notes
                    else:
                        notes = [slash_bass]
                    
                    # Rebuild midi array
                    midi = notes + [0] * (8 - len(notes))
                    chord_label = token  # Label is the bass note
            
            # NOW append exactly ONE chord for this DOT
            # Pad midi to 8 if needed
            while len(midi) < 8:
                midi.append(0)
            midi = midi[:8]
            
            # Only append if we have actual notes
            note_count = len([n for n in midi if n > 0])
            if note_count > 0:
                midi_sequence.append((midi.copy(), duration, chord_label))
                previous_voicing = [n for n in midi if n > 0]
        
            
        #Normalize the length of the MIDI sequence to 8 ----------------------------
        for i, item in enumerate(midi_sequence):    
            current_midi = item[0]
            item_duration = item[1]  # Use the item's actual duration, not the loop variable!
            element = item[2]
            if len(current_midi) < 8:
                for j in range(8 - len(current_midi)):
                    current_midi.append(0)
                # Update the item in the list with normalized MIDI
                midi_sequence[i] = (current_midi, item_duration, element)
     
        return midi_sequence, status
    
    
    #--------------------------------------------------------------------------------
    # Advanced voice leading methods inspired by modal_studio_Chord.js
    #--------------------------------------------------------------------------------
    
    def calculate_voice_leading_distance(self, prev_voicing, next_voicing):
        """
        Calculate total voice movement between two voicings.
        Lower values = better voice leading.
        
        This implements the voice leading principle: each note in a chord
        should move the minimum distance to its corresponding note in the next chord.
        
        Args:
            prev_voicing: List of MIDI note numbers from previous chord (non-zero notes only)
            next_voicing: List of MIDI note numbers for candidate next chord
            
        Returns:
            Total distance (sum of all voice movements in semitones)
        """
        if not prev_voicing or not next_voicing:
            return 0
        
        # Remove zeros from both voicings
        prev_notes = [n for n in prev_voicing if n != 0]
        next_notes = [n for n in next_voicing if n != 0]
        
        if not prev_notes or not next_notes:
            return 0
        
        total_distance = 0
        
        # For each note in the new chord, find its closest corresponding note in previous chord
        # This creates optimal voice leading
        used_prev_indices = set()
        
        for next_note in next_notes:
            min_distance = float('inf')
            best_prev_idx = 0
            
            # Try matching to each unused previous note
            for prev_idx, prev_note in enumerate(prev_notes):
                if prev_idx in used_prev_indices:
                    continue
                    
                # Calculate distance (considering octave equivalence)
                distance = abs(next_note - prev_note)
                
                # Also try octave transpositions to find closest version
                for octave_shift in [-12, 12]:
                    alt_distance = abs((next_note + octave_shift) - prev_note)
                    if alt_distance < distance:
                        distance = alt_distance
                
                if distance < min_distance:
                    min_distance = distance
                    best_prev_idx = prev_idx
            
            used_prev_indices.add(best_prev_idx)
            total_distance += min_distance
        
        return total_distance
    
    def select_best_voicing(self, previous_voicing, candidate_voicings, prev_root=None, curr_root=None):
        """
        Select best voicing from candidates based on voice leading efficiency.
        
        Args:
            previous_voicing: List of MIDI notes from previous chord
            candidate_voicings: List of pre-generated concrete voicings (Closed, Drop2, Drop3)
            prev_root: Ignored
            curr_root: Ignored
            
        Returns:
            The best voicing from the candidates list
        """
        if not previous_voicing or not candidate_voicings:
            return candidate_voicings[0] if candidate_voicings else [0, 4, 7, 12]
        
        prev_notes = [n for n in previous_voicing if n != 0]
        if not prev_notes:
            return candidate_voicings[0]
        
        best_voicing = None
        min_distance = float('inf')
        
        # Calculate center of previous voicing to bias towards keeping register
        prev_avg = sum(prev_notes) / len(prev_notes) if prev_notes else 60
        
        for voicing in candidate_voicings:
            # Filter out empty voicings
            if not voicing: continue
            
            # Calculate total voice movement distance
            # calculate_optimized_distance does a good job matching voice-to-voice
            distance = self.calculate_optimized_distance(prev_notes, voicing)
            
            # Add a small penalty for extreme register shifts (drift)
            # This helps keep the progression centered if movement is equal
            curr_avg = sum(voicing) / len(voicing)
            drift_penalty = abs(curr_avg - prev_avg) * 0.1
            
            total_score = distance + drift_penalty
            
            if total_score < min_distance:
                min_distance = total_score
                best_voicing = voicing
        
        return best_voicing if best_voicing is not None else candidate_voicings[0]
    
    def optimize_voicing_octaves(self, prev_notes, next_voicing):
        """
        Adjust octaves using Berklee voice leading principles.
        
        CRITICAL RULES:
        1. Root (bass note) stays FIXED
        2. Upper voices (3, 7, and any extensions) move to minimize distance
        3. Basic chord sound (3 and 7) should stay in comfortable range
        
        Args:
            prev_notes: List of MIDI notes from previous chord (non-zero only)
            next_voicing: Candidate voicing from template
            
        Returns:
            Voicing with octaves adjusted for smooth voice leading
        """
        next_notes = [n for n in next_voicing if n != 0]
        if not next_notes or not prev_notes:
            return next_voicing
        
        # ROOT STAYS FIXED in bass
        root = next_notes[0]
        upper_voices = next_notes[1:] if len(next_notes) > 1 else []
        
        # Get previous upper voices (exclude bass)
        prev_upper = sorted([n for n in prev_notes if n > prev_notes[0]])
        
        optimized = [root]  # Bass is locked
        
        # For each upper voice, find closest octave to previous upper voices
        for next_note in upper_voices:
            best_note = next_note
            min_distance = float('inf')
            
            # Try different octaves
            for octave_shift in [-24, -12, 0, 12, 24]:  # ±2 octaves
                candidate = next_note + octave_shift
                
                # Must be: in range, above root, in comfortable voicing range
                if candidate <= root or candidate < 24 or candidate > 96:
                    continue
                
                # For 3rd and 7th (typically first two upper voices), prefer range C3-C5
                # This follows Berklee recommendation for basic chord sound placement
                voice_idx = len(optimized) - 1
                if voice_idx <= 2:  # First two upper voices (likely 3 and 7)
                    if candidate < 48 or candidate > 72:  # Outside C3-C5
                        continue
                
                # Find distance to closest previous upper voice
                if prev_upper:
                    closest_prev_distance = min(abs(candidate - p) for p in prev_upper)
                else:
                    closest_prev_distance = abs(candidate - root)
                
                if closest_prev_distance < min_distance:
                    min_distance = closest_prev_distance
                    best_note = candidate
            
            optimized.append(best_note)
        
        # Sort upper voices only (keep root first)
        if len(optimized) > 1:
            upper_sorted = sorted(optimized[1:])
            optimized = [optimized[0]] + upper_sorted
        
        # Pad with zeros
        while len(optimized) < len(next_voicing):
            optimized.append(0)
        
        return optimized
    
    def check_and_add_ninth(self, prev_notes, current_notes, root_note):
        """
        Add 9th ONLY when it genuinely fills a gap in voice leading.
        
        Very conservative - only adds if there's a specific voice leading need.
        
        Args:
            prev_notes: Previous chord notes (non-zero)
            current_notes: Current optimized voicing (non-zero, sorted)
            root_note: Root note of current chord
            
        Returns:
            Voicing with 9th added only if truly needed
        """
        if not prev_notes or not current_notes or len(current_notes) < 3:
            return current_notes
        
        # Get highest notes
        prev_highest = max(prev_notes)
        current_highest = max(current_notes)
        
        # Only consider 9th if there's a big gap (>5 semitones) from prev to current highest
        gap = abs(prev_highest - current_highest)
        if gap <= 5:
            return current_notes  # No gap to fill
        
        # Try natural 9th first, then b9 only if natural doesn't work
        ninth_options = [2, 1]
        
        for ninth_interval in ninth_options:
            # Try one octave above root only
            ninth_note = root_note + ninth_interval + 12
            
            # Must be between prev and current highest (filling the gap)
            if not (min(prev_highest, current_highest) < ninth_note < max(prev_highest, current_highest)):
                continue
            
            # Must be in range and not too close to existing notes
            if ninth_note > 96:
                continue
                
            too_close = any(abs(ninth_note - n) <= 2 for n in current_notes)
            if too_close:
                continue
            
            # Found a 9th that fills the gap - use it and stop
            current_notes.append(ninth_note)
            current_notes.sort()
            break
        
        return current_notes
    
    def calculate_optimized_distance(self, prev_notes, next_notes):
        """
        Calculate total voice leading distance between two voicings.
        Each voice finds its nearest corresponding voice in the next chord.
        
        Args:
            prev_notes: Previous chord notes (non-zero only)
            next_notes: Next chord notes (already octave-optimized)
            
        Returns:
            Total distance (sum of minimum movements)
        """
        next_clean = [n for n in next_notes if n != 0]
        
        if not prev_notes or not next_clean:
            return 0
        
        # Greedy matching: each next note to its closest unused previous note
        used_prev = set()
        total_distance = 0
        
        # Sort both by pitch for better matching
        sorted_next = sorted(next_clean)
        sorted_prev = sorted(prev_notes)
        
        for next_note in sorted_next:
            min_dist = float('inf')
            best_prev_idx = 0
            
            for i, prev_note in enumerate(sorted_prev):
                if i in used_prev:
                    continue
                    
                dist = abs(next_note - prev_note)
                if dist < min_dist:
                    min_dist = dist
                    best_prev_idx = i
            
            used_prev.add(best_prev_idx)
            total_distance += min_dist
        
        return total_distance
    
    def select_voicing_by_position(self, position):
        """
        Select voicing template based on chord position in progression.
        Similar to selectVoicingBasedOnFunction in modal_studio_Chord.js
        
        Args:
            position: Position in sequence (0-6 maps to v_0 through v_6)
        
        Returns:
            Voicing key string ('v_0' through 'v_6')
        """
        voicing_index = position % 7
        return f'v_{voicing_index}'
    
    def optimize_voice_leading(self, current_voicing, next_voicing, root_current, root_next):
        """
        Optimize voice leading between two chords to minimize movement.
        Inspired by checkAndAddNinth and voice leading logic in modal_studio_Chord.js
        
        Args:
            current_voicing: List of MIDI notes for current chord
            next_voicing: List of MIDI notes for next chord  
            root_current: Root note of current chord
            root_next: Root note of next chord
            
        Returns:
            Optimized next_voicing with adjusted octaves for smooth voice leading
        """
        if not current_voicing or not next_voicing:
            return next_voicing
        
        # Work with non-zero notes only
        current_notes = [n for n in current_voicing if n != 0]
        next_notes = [n for n in next_voicing if n != 0]
        
        if not current_notes or not next_notes:
            return next_voicing
        
        # Optimize each voice to stay close to previous chord
        optimized = []
        for next_note in next_notes:
            # Find closest version of this note to any note in current chord
            best_note = next_note
            min_distance = float('inf')
            
            for current_note in current_notes:
                # Try this note and its octave transpositions
                for octave_shift in [-12, 0, 12]:
                    candidate = next_note + octave_shift
                    distance = abs(candidate - current_note)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_note = candidate
            
            optimized.append(best_note)
        
        # Pad with zeros to match original length
        while len(optimized) < len(next_voicing):
            optimized.append(0)
            
        return optimized
    
    def add_extensions_for_quality(self, voicing, chord_type, root):
        """
        Add chord extensions (9th, 11th, 13th) based on chord quality.
        Inspired by the upperNinth logic in modal_studio_Chord.js
        
        Args:
            voicing: Base voicing as list of intervals from root
            chord_type: Chord quality (e.g., 'maj7', 'm7', 'dom7')
            root: Root note MIDI number
            
        Returns:
            Extended voicing with appropriate color tones
        """
        extended = voicing.copy()
        
        # Add 9th for certain chord types (common in jazz)
        if chord_type in ['maj7', 'm7', 'dom7', 'sus7', 'm_maj7']:
            # Add 9th (14 semitones from root) in upper register if not too crowded
            if len(extended) < 6:
                ninth = 14  # 9th interval
                if ninth not in extended:
                    extended.append(ninth)
        
        # Add 11th for sus chords
        if chord_type in ['sus7', 'sus4']:
            if len(extended) < 6:
                eleventh = 17  # 11th interval  
                if eleventh not in extended:
                    extended.append(eleventh)
        
        # Add 13th for dominant and major chords
        if chord_type in ['dom7', 'maj7'] and len(extended) < 7:
            thirteenth = 21  # 13th interval
            if thirteenth not in extended:
                extended.append(thirteenth)
        
        return extended
    
    def get_drop_2_voicing(self, base_voicing):
        """
        Create a drop-2 voicing from a closed voicing.
        Drop-2 voicings are essential in jazz for better voice leading.
        
        Args:
            base_voicing: List of intervals in closed position
            
        Returns:
            Drop-2 voicing (second-highest note dropped an octave)
        """
        if len(base_voicing) < 3:
            return base_voicing
        
        drop2 = base_voicing.copy()
        # Drop the second note from the top down an octave
        if len(drop2) >= 2:
            drop2[-2] = drop2[-2] - 12
        
        # Re-sort to maintain ascending order
        drop2.sort()
        return drop2
    
    def get_drop_3_voicing(self, base_voicing):
        """
        Create a drop-3 voicing from a closed voicing.
        
        Args:
            base_voicing: List of intervals in closed position
            
        Returns:
            Drop-3 voicing (third-highest note dropped an octave)
        """
        if len(base_voicing) < 4:
            return base_voicing
        
        drop3 = base_voicing.copy()
        # Drop the third note from the top down an octave
        if len(drop3) >= 3:
            drop3[-3] = drop3[-3] - 12
        
        # Re-sort to maintain ascending order
        drop3.sort()
        return drop3
    
    
    #--------------------------------------------------------------------------------
    #Export the file to MIDI
    def export_to_midi(self, sequence, filename, path = "../dataset/midi_files/"):
        """
        Export sequence to MIDI file.
        
        SIMPLE: The sequence now contains ONLY chords (one per DOT).
        Each element is (midi_array, duration, label).
        Just export all of them in order.
        """
        # Create a MIDI file
        track    = 0
        channel  = 0
        tempo    = 120   # In BPM

        MyMIDI = MIDIFile()
        MyMIDI.addTempo(track, 0, tempo)

        time = 0  # Start time in beats
        
        for item in sequence:
            midi_notes = item[0]
            duration = float(item[1])
            
            # Get non-zero notes
            notes = [n for n in midi_notes if n > 0]
            
            # Skip if no notes
            if len(notes) == 0:
                continue
            
            # Convert duration from seconds to beats
            # At 120 BPM: 1 second = 2 beats
            duration_in_beats = duration * (tempo / 60.0)
            
            for pitch in notes:
                volume = int(random.uniform(55, 85))
                MyMIDI.addNote(track, channel, pitch, time, duration_in_beats, volume)
            
            time += duration_in_beats
  
        fullname = path + filename + '.mid'
        
        with open(fullname, "wb") as output_file:
            MyMIDI.writeFile(output_file)
        
        print('✓ MIDI file created:', filename + '.mid') 
        return fullname
        
        
    #--------------------------------------------------------------------------------
    #Get the chords from the sequence
    def get_chords(self, sequence):
        strings_array = [item[0] for item in sequence if item[0] != '']
        return strings_array
    
    #--------------------------------------------------------------------------------
    def play_midi(self, filename):
        # Check if the MIDI file exists
        if not os.path.exists(filename):
            print(f"Error: File not found - {filename}")
            return

         # Load the MIDI file into a music21 stream
        mf = midi.MidiFile()
        mf.open(filename)
        mf.read()
        mf.close()

        # Convert MIDI file to a music21 stream
        s = midi.translate.midiFileToStream(mf)

        # Play the MIDI stream
        s.show('midi')
        
    #--------------------------------------------------------------------------------
    #Convert the separated chords into one unify chord
    def convertChordsFromOutput(self, sequence):
        chord = []
        chordArray = []
        ignore = self.listToIgnore()
        ignore.remove('.') #we need the dot to identify the chord
        ignore.remove('|')
        ignore.remove(':|')
        ignore.remove('<end>')
        #if sequence[len(sequence)-1] != '.':
        #    sequence.append('.')
        for i in range (4, len(sequence)): #first four elements are style context
            element = sequence[i]
           
            if element not in ignore and element not in self.durations:
                if element == 'dom7':
                    element = '7'
                #check if the chord starts
                if element != '.' and element != '|' and element != ':|' and element != '<end>':
                    #print(i, duration)
                    #collect the elements of the chord
                    if element.find('add') >= 0 or element.find('subtract') >= 0 or element.find('alter') >= 0:
                        chord.append(' ')
                    chord.append(element)
                    #print(i, chord)
                if len(chord) > 0:
                    if  element == '.' or element == '|' or element == ':|' or element == '<end>':
                        #print(i, element)listToIgnore
                        #join the sections into a formatted chord
                        c = ''.join(chord) 
                        chordArray.append(c)
                        chord = []
                
        return chordArray
    #----------------------------------------------------
     # Function to create pitch bend messages
    def create_pitch_bend(self, value, channel):
        # Pitch bend value ranges from -8192 to 8191
        pitch = int((value / 100) * 8192)
        return Message('pitchwheel', pitch=pitch, channel=channel)
    
    #----------------------------------------------------
    def MidiChord(self, path = "../dataset/midi_files/", filename = 'detuned_Cmaj_chord'):
        
        # Create a new MIDI file and a new track
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Constants for note durations and volumes
        duration = 960  # duration in ticks (e.g., quarter note in standard MIDI)
        volume = 64  # Volume (0-127)

        # Define the MIDI note numbers for Cmaj7 chord
        C_note = 60  # Middle C
        E_note = 64  # E above middle C
        G_note = 67  # G above middle C
        B_note = 71  # B above middle C

        # MIDI channels for MPE
        C_channel = 0
        E_channel = 1
        G_channel = 2
        B_channel = 3


        # Time for the start of the chord
        start_time = 0

        # Add the C note to the MIDI file (no detuning)
        track.append(Message('note_on', note=C_note, velocity=volume, channel=C_channel, time=start_time))

        # Add the detuned E note
        track.append(self.create_pitch_bend(-25, E_channel))  # -25 cents detune
        track.append(Message('note_on', note=E_note, velocity=volume, channel=E_channel, time=start_time))

        # Add the G note to the MIDI file (no detuning)
        track.append(Message('note_on', note=G_note, velocity=volume, channel=G_channel, time=start_time))

        # Add the detuned B note
        track.append(self.create_pitch_bend(-28, B_channel))  # -28 cents detune
        track.append(Message('note_on', note=B_note, velocity=volume, channel=B_channel, time=start_time))

        # Add note off messages for all notes at the same time (duration ticks later)
        track.append(Message('note_off', note=C_note, velocity=volume, channel=C_channel, time=duration))
        track.append(Message('note_off', note=E_note, velocity=volume, channel=E_channel, time=0))  # time=0 because it’s the same moment
        track.append(Message('note_off', note=G_note, velocity=volume, channel=G_channel, time=0))
        track.append(Message('note_off', note=B_note, velocity=volume, channel=B_channel, time=0))
        track.append(self.create_pitch_bend(0, E_channel))  # Reset pitch bend for E
        track.append(self.create_pitch_bend(0, B_channel))  # Reset pitch bend for B

        tz = pytz.timezone('Europe/Stockholm')
        stockholm_now = datetime.now(tz)
        mh = str(stockholm_now.hour)
        mm = str(stockholm_now.minute)
        ms = str(stockholm_now.second)
        
        if len(mh) == 1:
            mh = '0' + str(stockholm_now.hour)
        if len(mm) == 1:
            mm= '0' + str(stockholm_now.minute)
        if len(ms) == 1:
            ms = '0' + str(stockholm_now.second)
            
        ext =  mh + mm + ms + '_' +str(stockholm_now.day) + '_' + str(stockholm_now.month) + '_' + str(stockholm_now.year) + '_'
        
        fullname = path + ext + filename + '.mid'
        currentName = ext + filename + '.mid'
        
        # Save the MIDI file
        mid.save(fullname)
            
        print("MIDI file generated: ", currentName)


    
   
        
    
      