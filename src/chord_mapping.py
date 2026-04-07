
# 53-TET Chord Mapping Definitions

# 53-TET Note Names (Indices 0 to 52)
# Based on user provided list:
# C, ^C, ^^C, vvC#, vC#, C#, ^C#, ^^C#, vD, D, ^D, ^^D, vvD#, vD#, D#, ^^Eb, vvE, vE, E, ^E, ^^E, vF, F, ^F, ^^F, vvF#, vF#, F#, ^F#, ^^F#, vG, G, ^G, ^^G, vvG#, vG#, G#, ^G#, vvA, vA, A, ^A, ^^A, vBb, Bb, ^Bb, ^^Bb, vvB, vB, B, ^B, ^^B, vC
# Note: The last "vC" might be the 53rd step (wrapping to C) or index 52. 
# Usually 53-TET has 53 unique pitches. 
# Let's count them:
# 1.C 2.^C 3.^^C 4.vvC# 5.vC# 6.C# 7.^C# 8.^^C# 9.vD 10.D 11.^D 12.^^D 13.vvD# 14.vD# 15.D# 16.^^Eb 
# 17.vvE 18.vE 19.E 20.^E 21.^^E 22.vF 23.F 24.^F 25.^^F 26.vvF# 27.vF# 28.F# 29.^F# 30.^^F# 31.vG 
# 32.G 33.^G 34.^^G 35.vvG# 36.vG# 37.G# 38.^G# 39.vvA 40.vA 41.A 42.^A 43.^^A 44.vBb 45.Bb 46.^Bb 
# 47.^^Bb 48.vvB 49.vB 50.B 51.^B 52.^^B 53.vC
# That is 53 items exactly.

NOTE_NAMES_53TET = [
    "C", "^C", "^^C", "vvC#", "vC#", "C#", "^C#", "^^C#", "vD", "D", "^D", "^^D", "vvD#", "vD#", "D#", "^^Eb",
    "vvE", "vE", "E", "^E", "^^E", "vF", "F", "^F", "^^F", "vvF#", "vF#", "F#", "^F#", "^^F#", "vG",
    "G", "^G", "^^G", "vvG#", "vG#", "G#", "^G#", "vvA", "vA", "A", "^A", "^^A", "vBb", "Bb", "^Bb",
    "^^Bb", "vvB", "vB", "B", "^B", "^^B", "vC"
]

# Map 12-TET intervals to note offsets (approx) for chord construction
# Root, m3, M3, P5, m7, M7
# We need to identify chords in the text file. 
# Common types: "maj" (M3, P5), "m" (m3, P5), "maj7" (M3, P5, M7), "m7" (m3, P5, m7), "dom7" (M3, P5, m7)

CHORD_STRUCTURES_12TET = {
    'maj': [0, 4, 7],
    'major': [0, 4, 7],
    'm': [0, 3, 7],
    'min': [0, 3, 7],
    'minor': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    '7': [0, 4, 7, 10],
    'dim7': [0, 3, 6, 9],
    'm7b5': [0, 3, 6, 10],
    'sus4': [0, 5, 7],
    'sus2': [0, 2, 7],
    '6': [0, 4, 7, 9],
    'm6': [0, 3, 7, 9],
}

# The definitions of "Qualities" in 53 terms (Step Count)
# This logic is inferred from known 53-TET intervals and the table provided.
# Keys are 'interval_class' (3rd, 5th, 7th).
# We assume "Perfect 5th" is always ~31 steps.
# "Major 3rd" is ~18 steps.
# "Minor 3rd" is ~13-14 steps.

INTERVAL_STEPS_TO_QUALITY = {
    # Thirds (Range approx 10-22)
    10: "subminor", # Guess
    11: "subminor",
    12: "subminor", # Confirmed by Type 2
    13: "minor",
    14: "neutral-minor",
    15: "neutral", # Confirmed by Type 1
    16: "neutral-major",
    17: "downmajor",
    18: "major", # Confirmed by Type 0
    19: "upmajor",
    20: "supermajor",
    21: "supermajor", 
    22: "perfect", # 4th

    # Fifths (Range approx 28-34)
    30: "diminished", # Wolf 5th?
    31: "perfect",
    32: "augmented",

    # Sevenths (Range approx 40-52)
    40: "subminor", # Guess
    41: "subminor",
    42: "subminor",
    43: "downminor", # ??
    44: "minor", # 53-9=44?
    45: "neutral-minor",
    46: "neutral", # 53-7=46?
    47: "neutral-major",
    48: "downmajor",
    49: "major", # 53-4=49?
    50: "upmajor",
    51: "supermajor",
    52: "supermajor"
}
# Note: this dictionary is a lookup helper. We will try to find exact matches.
# If not found, we pick the closest.

# The Lookup Table for New Chord Names
# Keys: (Third_Qual, Fifth_Qual, Seventh_Qual)
# Value: Suffix (e.g., "S^S^7")

CHORD_NAMING_TABLE = {
    ("supermajor", "perfect", "supermajor"): "S^S^7",
    ("supermajor", "perfect", "upmajor"): "S^^7",
    ("supermajor", "perfect", "major"): "S^M7",
    ("supermajor", "perfect", "downmajor"): "S^^7", # Note: User listed S^^7 twice? Or typo in user request "S^^7" for downmajor? Checking request.. "downmajor -> S^^7". "upmajor -> S^^7". Suspicious.
    # Re-reading user request:
    # supermajor | perfect | upmajor | S^^7
    # supermajor | perfect | major | S^M7
    # supermajor | perfect | downmajor | S^^7  <-- User data repeats S^^7? Wait. 
    # Maybe "S^v7"? Typos are possible. I'll stick to data provided strictly.
    
    ("supermajor", "perfect", "neutral-major"): "S^NM7",
    ("supermajor", "perfect", "neutral"): "S^N7",
    ("supermajor", "perfect", "neutral-minor"): "S^Nm7",
    ("supermajor", "perfect", "minor"): "S^m7",
    ("supermajor", "perfect", "downminor"): "S^vm7",
    ("supermajor", "perfect", "subminor"): "S^sm7",

    ("upmajor", "perfect", "supermajor"): "^S^7",
    ("upmajor", "perfect", "upmajor"): "^^7",
    ("upmajor", "perfect", "major"): "^M7",
    ("upmajor", "perfect", "downmajor"): "^v7",
    ("upmajor", "perfect", "neutral-major"): "^NM7",
    ("upmajor", "perfect", "neutral"): "^N7",
    ("upmajor", "perfect", "neutral-minor"): "^Nm7",
    ("upmajor", "perfect", "minor"): "^m7",
    ("upmajor", "perfect", "downminor"): "^vm7",
    ("upmajor", "perfect", "subminor"): "^sm7",

    ("major", "perfect", "supermajor"): "S^7",
    ("major", "perfect", "upmajor"): "^M7",
    ("major", "perfect", "major"): "maj7",
    ("major", "perfect", "downmajor"): "vM7",
    ("major", "perfect", "neutral-major"): "NM7",
    ("major", "perfect", "neutral"): "N7",
    ("major", "perfect", "neutral-minor"): "Nm7",
    ("major", "perfect", "minor"): "7",
    ("major", "perfect", "downminor"): "vm7",
    ("major", "perfect", "subminor"): "sm7",

    ("downmajor", "perfect", "supermajor"): "vMS^7",
    ("downmajor", "perfect", "upmajor"): "vM^M7",
    ("downmajor", "perfect", "major"): "vMM7",
    ("downmajor", "perfect", "downmajor"): "vMvM7",
    ("downmajor", "perfect", "neutral-major"): "vMNM7",
    ("downmajor", "perfect", "neutral"): "vMN7",
    ("downmajor", "perfect", "neutral-minor"): "vMNm7",
    ("downmajor", "perfect", "minor"): "vMm7",
    ("downmajor", "perfect", "downminor"): "vMvm7",
    ("downmajor", "perfect", "subminor"): "vMsm7",

    ("neutral-major", "perfect", "supermajor"): "NMS^7",
    ("neutral-major", "perfect", "upmajor"): "NM^7",
    ("neutral-major", "perfect", "major"): "NMM7",
    ("neutral-major", "perfect", "downmajor"): "NMvM7",
    ("neutral-major", "perfect", "neutral-major"): "NMNM7",
    ("neutral-major", "perfect", "neutral"): "NMN7",
    ("neutral-major", "perfect", "neutral-minor"): "NMNm7",
    ("neutral-major", "perfect", "minor"): "NMm7",
    ("neutral-major", "perfect", "downminor"): "NMvm7",
    ("neutral-major", "perfect", "subminor"): "NMsm7",

    ("neutral", "perfect", "supermajor"): "NS^7",
    ("neutral", "perfect", "upmajor"): "N^7",
    ("neutral", "perfect", "major"): "NM7",
    ("neutral", "perfect", "downmajor"): "NvM7",
    ("neutral", "perfect", "neutral-major"): "NNM7",
    ("neutral", "perfect", "neutral"): "NN7",
    ("neutral", "perfect", "neutral-minor"): "NNm7",
    ("neutral", "perfect", "minor"): "Nm7",
    ("neutral", "perfect", "downminor"): "Nvm7",
    ("neutral", "perfect", "subminor"): "Nsm7",

    ("neutral-minor", "perfect", "supermajor"): "NmS^7",
    ("neutral-minor", "perfect", "upmajor"): "Nm^7",
    ("neutral-minor", "perfect", "major"): "NmM7",
    ("neutral-minor", "perfect", "downmajor"): "NmvM7",
    ("neutral-minor", "perfect", "neutral-major"): "NmNM7",
    ("neutral-minor", "perfect", "neutral"): "NmN7",
    ("neutral-minor", "perfect", "neutral-minor"): "NmNm7",
    ("neutral-minor", "perfect", "minor"): "Nmm7",
    ("neutral-minor", "perfect", "downminor"): "Nmvm7",
    ("neutral-minor", "perfect", "subminor"): "Nmsm7",

    ("minor", "perfect", "supermajor"): "mS^7",
    ("minor", "perfect", "upmajor"): "m^7",
    ("minor", "perfect", "major"): "mM7",
    ("minor", "perfect", "downmajor"): "mvM7",
    ("minor", "perfect", "neutral-major"): "mNM7",
    ("minor", "perfect", "neutral"): "mN7",
    ("minor", "perfect", "neutral-minor"): "mNm7",
    ("minor", "perfect", "minor"): "m7",
    ("minor", "perfect", "downminor"): "mvm7",
    ("minor", "perfect", "subminor"): "msm7",

    ("downminor", "perfect", "supermajor"): "vmS^7",
    ("downminor", "perfect", "upmajor"): "vm^7",
    ("downminor", "perfect", "major"): "vmM7",
    ("downminor", "perfect", "downmajor"): "vmvM7",
    ("downminor", "perfect", "neutral-major"): "vmNM7",
    ("downminor", "perfect", "neutral"): "vmN7",
    ("downminor", "perfect", "neutral-minor"): "vmNm7",
    ("downminor", "perfect", "minor"): "vmm7",
    ("downminor", "perfect", "downminor"): "vmvm7",
    ("downminor", "perfect", "subminor"): "vmsm7",

    ("subminor", "perfect", "subminor"): "smsm7",
    ("subminor", "perfect", "minor"): "smm7",
    ("subminor", "perfect", "neutral"): "smN7",
    ("subminor", "perfect", "major"): "smM7",
    ("subminor", "perfect", "supermajor"): "smS^7", # super-major in table
    
    # Diminished 5ths
    ("minor", "diminished", "subminor"): "smø7",
    ("minor", "diminished", "minor"): "ø7",
    ("minor", "diminished", "neutral"): "øN7",
    ("minor", "diminished", "major"): "øM7",
    ("minor", "diminished", "supermajor"): "øS^7", # super-major

    # Augmented 5ths
    ("major", "augmented", "subminor"): "M+sm7",
    ("major", "augmented", "minor"): "M+m7",
    ("major", "augmented", "neutral"): "M+N7",
    ("major", "augmented", "major"): "M+7",
    ("major", "augmented", "supermajor"): "M+S^7", # super-major
}

