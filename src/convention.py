
# 53-TET Chord Naming Convention
# This file governs how intervals are translated into chord symbols.

# -----------------------------------------------------------------------------
# 1. Semantic Quality Definitions
# -----------------------------------------------------------------------------

STEP_TO_SEMANTIC = {
    # Thirds
    9: "major-second", # Standard Major 2nd (2 semitones ~ 8.8)
    10: "augmented-second", # or subminor third?
    11: "subminor",
    12: "downminor",
    13: "minor",
    14: "neutral-minor",
    15: "neutral",
    16: "neutral-major",
    17: "downmajor",
    18: "major",
    19: "upmajor",
    20: "supermajor",
    21: "ultramajor",
    
    # Fifths
    22: "fourth",          # Perfect 4th (Standard 5 semitones = ~22)
    26: "diminished-fifth",# Standard Diminished 5th (6 semitones = ~26.5)
    28: "subdiminished",
    29: "downdiminished",
    30: "diminished", 
    31: "perfect",
    32: "augmented",
    33: "upaugmented",
    35: "augmented-fifth", # Standard Augmented 5th (8 semitones = ~35.3)
    
    # Sevenths
    39: "major-sixth",     # Standard Major 6th (9 semitones = ~39.7)
    40: "diminished-seventh", # Standard Dim 7th (Enharmonically Major 6th)
    42: "subminor",
    43: "downminor",
    44: "minor",
    45: "neutral-minor",
    46: "neutral",
    47: "neutral-major",
    48: "downmajor",
    49: "major",
    50: "upmajor",
    51: "supermajor"
}

# -----------------------------------------------------------------------------
# 2. Strict Abbreviation Mappings
# -----------------------------------------------------------------------------

# Base Triad Prefixes (derived from the Third)
TRIAD_PREFIXES = {
    "supermajor": "S^",
    "ultramajor": "S^^",
    "upmajor": "^",
    "major": "",       # Standard Major is empty (e.g., "C")
    "downmajor": "vM",
    "neutral-major": "NM",
    "neutral": "N",
    "neutral-minor": "Nm",
    "minor": "m",
    "downminor": "vm",
    "subminor": "sm"
}

# Fifth Suffixes
FIFTH_SUFFIXES = {
    "perfect": "",
    "diminished": "dim",    # The comma-flat fifth (30)
    "augmented": "+",       # The comma-sharp fifth (32)
    "diminished-fifth": "(b5)", # Standard Diminished (26)
    "augmented-fifth": "(#5)",  # Standard Augmented (35)
    "major-second": "(sus2)",
    "fourth": "(sus4)",
    "downdiminished": "vdim",
    "subdiminished": "sdim",
    "upaugmented": "^+"
}

# Seventh Suffixes
# These are appended to the base. 
# Note: Context matters (e.g., Major Triad + Minor 7th = "7", but Minor Triad + Minor 7th = "7")
SEVENTH_SUFFIXES = {
    "supermajor": "S^7",
    "upmajor": "^7",    # Check table: sometimes just "^"?
    "major": "M7",
    "downmajor": "vM7",
    "neutral-major": "NM7",
    "neutral": "N7",
    "neutral-minor": "Nm7",
    "minor": "7",       # The dominant 7th interval
    "downminor": "vm7",
    "subminor": "sm7",
    "diminished-seventh": "dim7",
    "major-sixth": "6"
}

# -----------------------------------------------------------------------------
# 3. The Lookup Table (Source of Truth)
# -----------------------------------------------------------------------------

CHORD_RULES_TABLE = [
    # (Third, Fifth, Seventh, Name)
    ("supermajor", "perfect", "supermajor", "S^S^7"),
    ("supermajor", "perfect", "upmajor", "S^^7"),
    ("supermajor", "perfect", "major", "S^M7"),
    ("supermajor", "perfect", "downmajor", "S^^7"),
    ("supermajor", "perfect", "neutral-major", "S^NM7"),
    ("supermajor", "perfect", "neutral", "S^N7"),
    ("supermajor", "perfect", "neutral-minor", "S^Nm7"),
    ("supermajor", "perfect", "minor", "S^m7"),
    ("supermajor", "perfect", "downminor", "S^vm7"),
    ("supermajor", "perfect", "subminor", "S^sm7"),
    
    ("upmajor", "perfect", "supermajor", "^S^7"),
    ("upmajor", "perfect", "upmajor", "^^7"),
    ("upmajor", "perfect", "major", "^M7"),
    ("upmajor", "perfect", "downmajor", "^v7"),
    ("upmajor", "perfect", "neutral-major", "^NM7"),
    ("upmajor", "perfect", "neutral", "^N7"),
    ("upmajor", "perfect", "neutral-minor", "^Nm7"),
    ("upmajor", "perfect", "minor", "^m7"),
    ("upmajor", "perfect", "downminor", "^vm7"),
    ("upmajor", "perfect", "subminor", "^sm7"),
    
    ("major", "perfect", "supermajor", "S^7"),
    ("major", "perfect", "upmajor", "^M7"), 
    ("major", "perfect", "major", "maj7"),
    ("major", "perfect", "downmajor", "vM7"),
    ("major", "perfect", "neutral-major", "NM7"),
    ("major", "perfect", "neutral", "N7"),
    ("major", "perfect", "neutral-minor", "Nm7"),
    ("major", "perfect", "minor", "7"),
    ("major", "perfect", "downminor", "vm7"),
    ("major", "perfect", "subminor", "sm7"),
    
    ("downmajor", "perfect", "supermajor", "vMS^7"),
    ("downmajor", "perfect", "upmajor", "vM^M7"),
    ("downmajor", "perfect", "major", "vMM7"),
    ("downmajor", "perfect", "downmajor", "vMvM7"),
    ("downmajor", "perfect", "neutral-major", "vMNM7"),
    ("downmajor", "perfect", "neutral", "vMN7"),
    ("downmajor", "perfect", "neutral-minor", "vMNm7"),
    ("downmajor", "perfect", "minor", "vMm7"),
    ("downmajor", "perfect", "downminor", "vMvm7"),
    ("downmajor", "perfect", "subminor", "vMsm7"),
    
    ("neutral-major", "perfect", "supermajor", "NMS^7"),
    ("neutral-major", "perfect", "upmajor", "NM^7"),
    ("neutral-major", "perfect", "major", "NMM7"),
    ("neutral-major", "perfect", "downmajor", "NMvM7"),
    ("neutral-major", "perfect", "neutral-major", "NMNM7"),
    ("neutral-major", "perfect", "neutral", "NMN7"),
    ("neutral-major", "perfect", "neutral-minor", "NMNm7"),
    ("neutral-major", "perfect", "minor", "NMm7"),
    ("neutral-major", "perfect", "downminor", "NMvm7"),
    ("neutral-major", "perfect", "subminor", "NMsm7"),
    
    ("neutral", "perfect", "supermajor", "NS^7"),
    ("neutral", "perfect", "upmajor", "N^7"),
    ("neutral", "perfect", "major", "NM7"),
    ("neutral", "perfect", "downmajor", "NvM7"),
    ("neutral", "perfect", "neutral-major", "NNM7"),
    ("neutral", "perfect", "neutral", "NN7"),
    ("neutral", "perfect", "neutral-minor", "NNm7"),
    ("neutral", "perfect", "minor", "Nm7"),
    ("neutral", "perfect", "downminor", "Nvm7"),
    ("neutral", "perfect", "subminor", "Nsm7"),
    
    ("neutral-minor", "perfect", "supermajor", "NmS^7"),
    ("neutral-minor", "perfect", "upmajor", "Nm^7"),
    ("neutral-minor", "perfect", "major", "NmM7"),
    ("neutral-minor", "perfect", "downmajor", "NmvM7"),
    ("neutral-minor", "perfect", "neutral-major", "NmNM7"),
    ("neutral-minor", "perfect", "neutral", "NmN7"),
    ("neutral-minor", "perfect", "neutral-minor", "NmNm7"),
    ("neutral-minor", "perfect", "minor", "Nmm7"),
    ("neutral-minor", "perfect", "downminor", "Nmvm7"),
    ("neutral-minor", "perfect", "subminor", "Nmsm7"),
    
    ("minor", "perfect", "supermajor", "mS^7"),
    ("minor", "perfect", "upmajor", "m^7"),
    ("minor", "perfect", "major", "mM7"),
    ("minor", "perfect", "downmajor", "mvM7"),
    ("minor", "perfect", "neutral-major", "mNM7"),
    ("minor", "perfect", "neutral", "mN7"),
    ("minor", "perfect", "neutral-minor", "mNm7"),
    ("minor", "perfect", "minor", "m7"),
    ("minor", "perfect", "downminor", "mvm7"),
    ("minor", "perfect", "subminor", "msm7"),
    
    ("downminor", "perfect", "supermajor", "vmS^7"),
    ("downminor", "perfect", "upmajor", "vm^7"),
    ("downminor", "perfect", "major", "vmM7"),
    ("downminor", "perfect", "downmajor", "vmvM7"),
    ("downminor", "perfect", "neutral-major", "vmNM7"),
    ("downminor", "perfect", "neutral", "vmN7"),
    ("downminor", "perfect", "neutral-minor", "vmNm7"),
    ("downminor", "perfect", "minor", "vmm7"),
    ("downminor", "perfect", "downminor", "vmvm7"),
    ("downminor", "perfect", "subminor", "vmsm7"),
    
    ("subminor", "perfect", "super-major", "smS^7"), 
    ("subminor", "perfect", "subminor", "smsm7"),
    ("subminor", "perfect", "minor", "smm7"),
    ("subminor", "perfect", "neutral", "smN7"),
    ("subminor", "perfect", "major", "smM7"),
    
    # Diminished Families
    ("minor", "diminished", "subminor", "smø7"),
    ("minor", "diminished", "minor", "ø7"),
    ("minor", "diminished", "neutral", "øN7"),
    ("minor", "diminished", "major", "øM7"),
    ("minor", "diminished", "super-major", "øS^7"),
    
    # Augmented Families
    ("major", "augmented", "subminor", "M+sm7"),
    ("major", "augmented", "minor", "M+m7"),
    ("major", "augmented", "neutral", "M+N7"),
    ("major", "augmented", "major", "M+7"),
    ("major", "augmented", "super-major", "M+S^7"),
    
    # Explicit Triads (Optional but helpful for explicit handling)
    ("major", "perfect", None, ""),
    ("minor", "perfect", None, "m"),
    ("diminished", "diminished-fifth", None, "dim") # Assuming diminished 3rd exists? No, usually minor 3rd.
]

def get_name(q3, q5, q7):
    """
    Retrieves the chord name based on semantic qualities.
    If exact match not found, constructs it using strict abbreviations.
    """
    
    # 1. Normalize inputs
    if q7: q7 = q7.replace("super-major", "supermajor")
    
    # Alias handling for 5th (Treat diminished-fifth as diminished for table lookup)
    lookup_q5 = q5
    if q5 == "diminished-fifth":
         lookup_q5 = "diminished"
    
    # 2. Try Table Lookup
    if q7 is not None:
        for r3, r5, r7, name in CHORD_RULES_TABLE:
            # Check 5th with alias
            if r3 == q3 and (r5 == q5 or r5 == lookup_q5) and r7 == q7:
                return name
            
            # Fallback for hyphen mismatch
            r7_norm = r7.replace("-", "") if r7 else ""
            q7_norm = q7.replace("-", "") if q7 else ""
            if r3 == q3 and (r5 == q5 or r5 == lookup_q5) and r7_norm == q7_norm:
                return name
    else:
        # Triad Lookup
        for r3, r5, r7, name in CHORD_RULES_TABLE:
            if r7 is None and r3 == q3 and (r5 == q5 or r5 == lookup_q5):
                return name

    # 3. Construct Logic (Strict Abbreviation)
    
    # Base
    base = TRIAD_PREFIXES.get(q3, f"({q3})")
    
    # Fifth
    fifth = FIFTH_SUFFIXES.get(q5, "")
    if fifth == "" and q5 not in FIFTH_SUFFIXES and q5 is not None:
         if q5 != "perfect":
             fifth = f"({q5})"
    
    # Sevenths
    if q7 is None:
        result = f"{base}{fifth}"
        # Ensure we return empty string if result is empty, not None
        if not result: return ""
        return result
        
    seventh = SEVENTH_SUFFIXES.get(q7, "")
    
    # Contextual Logic for Minor 7th
    if q7 == "minor":
        if q3 == "major":
            seventh = "7"
        elif base.endswith("m"): 
            seventh = "7"
        else:
            seventh = "m7" 

    # Contextual Logic for Diminished 7th (Full Diminished)
    if q7 == "diminished-seventh":
        if "dim" in fifth or fifth == "(b5)":
             # imply dim7 for full dim
             if base == "m" and (q5 == "diminished" or q5 == "diminished-fifth"):
                  return "dim7"
                  
             seventh = "7"
             
        elif fifth == "":
             seventh = "dim7" 

    return f"{base}{fifth}{seventh}"
