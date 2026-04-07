[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# Microtonal Data Formation

[ANIMA](https://cordis.europa.eu/project/id/101203318) (Artificial INtelligence-based Interactive Microtonal Compositional Assistant) 
This is a pipeline for creating hybrid 12-TET/53-TET MIDI chord progression datasets for training transformer models on microtonal harmony.


## 🎯 Project Overview

This project aims to create a comprehensive dataset of chord progressions that bridges standard Western harmony (12-tone equal temperament) with microtonal music (53-TET) for training GPT-2 style models capable of generating musically coherent microtonal compositions.

**Source**: ~4,000 jazz standards from iReal Pro  
**Target**: 48,000+ transposed progressions with microtonal augmentation  
**Output**: MPE-MIDI format with pitch bend for microtonal accuracy

---

## 📋 Pipeline Stages

### **Stage 1: iReal → MIDI Dataset Creation**

Convert iReal Pro chord charts to high-quality MIDI with professional voicing.

#### Voicing Strategy
- **Register**: Bass root in C2-C3, chord tones spanning 1-2 octaves
- **Voicing Style**: 7 distinct voicing templates per chord type:
  - Various open and closed positions
  - Different note spacings and doublings
  - Extended voicings with 9ths, 11ths, 13ths
- **Voice Leading**: Smooth transitions with minimal motion between chords
- **Duration**: Block chords (whole/half notes) to focus on harmonic content

#### Rhythm Implementation
- **Simple Approach**: Quarter/half note chords aligned to harmonic rhythm

#### Technical Implementation
Enhanced `voicing.py` module with 7 voicing templates (`v_0` through `v_6`) per chord type:
- Each template offers different harmonic textures and voice distributions
- Automated template selection based on chord position in progression
- Support for all common jazz chord types (maj7, m7, dom7, ø7, dim7, sus, aug, etc.)
- Voice leading optimization methods available

#### What's Been Completed (Notebook: 01_musicXML_parser.ipynb)
1. ✅ **XML Parsing** - Parse ~4,000 iReal Pro XML files into structured chord progressions
2. ✅ **Song Structure Expansion** - Expand repeats, codas, and form markers into full sequences
3. ✅ **Duration Handling** - Extract and process rhythmic durations from XML
4. ✅ **MIDI Voicing** - Convert chord symbols to MIDI note arrays using voicing.py
5. ✅ **Validation** - XML-to-token accuracy verification (~93.4% match rate)

**Status**: ✅ Stage 1 Core Complete - Ready for Stage 2 (Transposition)

---

### **Stage 2: Enharmonic Transposition**

Expand dataset through intelligent transposition.

#### Augmentation Strategy
- **Scale**: 12 transpositions per song → **48,000 examples**
- **Key Consideration**: Track actual pitch height (critical for 53-TET mapping)
  - In 12-TET: C# = Db
  - In 53-TET: C# ≠ Db (different microtonal positions)
- **Register Management**: Avoid extremely high/low transpositions

#### Benefits
- Natural key distribution balance
- Model generalization across all keys
- Manageable dataset size for training

#### Implementation Progress
- ✅ **Transposition Module** - `transposition.py` with `transpose_song()` method
- ✅ **Testing** - Verified transposition on sample songs
- ⏳ **Full Dataset** - Need to run complete 12-key augmentation on all 4,000 songs

**Status**: 🔧 In Progress - Transposition code ready, needs full dataset run

---

### **Stage 3: Microtonal Data Augmentation**

Progressive introduction of 53-TET microtonality.

#### Three Levels of Microtonal Integration

##### **Level 1: 10% Microtonal (Sparse Substitutions)**
- Replace 1-2 chords per progression with 53-TET alternatives
- **Targets**: Dominant chords (septimal 7ths), color chords (maj7, min7)
- **Goal**: Teach model "microtonal chords in familiar contexts"
- **Method**: Maintain functional harmonic logic

##### **Level 2: 50% Microtonal (Hybrid)**
- Systematic alternation: 12-TET → 53-TET → 12-TET → 53-TET
- **Goal**: Model learns transitions between tuning systems
- **Application**: Bridges familiar and novel harmonic spaces

##### **Level 3: 100% Microtonal (Full EigenSpace)**
- Entire progressions in 53-TET
- Navigate EigenSpace using dissonance metrics
- **Goal**: Purely microtonal harmonic syntax
- **Application**: Explore novel microtonal progressions

#### 53-TET Substitution Methods

| Method    | Approach | Best For |
|-----------|----------|----------|
| 1 | Pre-map 12-TET → 53-TET equivalents<br/>| Adding familiar sonorities |
| 2 | EigenSpace dissonance distance to move the harmony into other chords | Exploring new harmonic space |
| 3 | Add intermediate chords | (preserves function) |

**Status**: 🔄 Planned

---

### **Stage 4: Tokenization Strategy**

Convert MIDI data to transformer-compatible token sequences with metadata prefixes.

#### Sequence Structure

Each sequence consists of two parts:
1. **Metadata Header** - Musical context and properties
2. **MIDI Content** - Note events with pitch bend information

#### Hybrid Vocabulary Design

##### Metadata Tokens (Sequence Prefix)
- `<SONG_NAME>` - Song title or identifier
- `<STYLE>` - Genre/style (jazz, bossa, swing, ballad, etc.)
- `<KEY>` - Tonal center (C, Bb, F#m, etc.)
- `<TEMPO>` - BPM value (60-240)
- `<TIME_SIG>` - Time signature (4/4, 3/4, 5/4, etc.)
- `<FORM>` - Song structure (AABA, ABAC, 12-bar blues, etc.)
- `<TUNING>` - Tuning system (12-TET, 53-TET-10%, 53-TET-50%, 53-TET-100%)
- `<BARS>` - Total number of bars

##### Core MIDI Token Types
- **Pitch**: 0-127 (standard MIDI, represents 12-TET base)
- **Pitch Bend**: Discrete values (-100 to +100 in 2¢ steps ≈ 100 tokens)
- **Time**: Quantized to 16th or 32nd notes
- **Channel**: 1-15 (MPE channels for polyphonic pitch bend)
- **Velocity**: Quantized to 8-16 levels

##### Structural Tokens
- `<BOS>`, `<EOS>` - Sequence boundaries
- `<BAR>`, `<BEAT>` - Metrical structure
- `<SECTION>` - Form sections (A, B, C, Bridge, Coda, etc.)

#### Example Token Sequence (Note-Level with Metadata)
```
<BOS>
<SONG_NAME_Autumn_Leaves>
<STYLE_jazz_standard>
<KEY_Gm>
<TEMPO_120>
<TIME_SIG_4/4>
<FORM_AABA>
<TUNING_53-TET-10%>
<BARS_32>

<SECTION_A> <BAR_1>
<TIME_0> <NOTE_ON_55_ch2_v80> <BEND_ch2_+0>    # Root: G
<TIME_0> <NOTE_ON_58_ch3_v75> <BEND_ch3_-14>   # Minor 3rd (microtonal)
<TIME_0> <NOTE_ON_62_ch4_v72> <BEND_ch4_+0>    # Perfect 5th
<TIME_480> <NOTE_OFF_55_ch2> <NOTE_OFF_58_ch3> <NOTE_OFF_62_ch4>

<BAR_2>
<TIME_0> <NOTE_ON_60_ch2_v80> <BEND_ch2_+0>    # Next chord...
...
<EOS>
```

#### Alternative: Compound Chord Tokens with Metadata
```
<BOS>
<SONG_NAME_All_The_Things_You_Are>
<STYLE_jazz_ballad>
<KEY_Ab>
<TEMPO_80>
<TUNING_12-TET>

<SECTION_A> <BAR_1>
<CHORD root=Ab type=maj7 voices=[56,60,63,67] bends=[0,0,0,0] dur=1920>
<BAR_2>
<CHORD root=F type=m7 voices=[53,57,60,64] bends=[0,0,0,0] dur=1920>
...
<EOS>
```

#### Benefits of Metadata Prefix
- **Conditioning**: Model can generate in specific styles, keys, or tuning systems
- **Analysis**: Easy filtering and analysis of generated outputs by metadata
- **Controllable Generation**: Users can specify desired characteristics
- **Context**: Provides harmonic and stylistic context before processing notes

**Trade-off**: Slightly longer sequences, but dramatically improves controllability

**Status**: 🔄 Planned

---

### **Stage 5: GPT-2 Model Training**

Train transformer model on hybrid 12-TET/53-TET sequences.

#### Architecture
- **Model Size**: GPT-2 Small (124M parameters) - sufficient for this domain
- **Context Window**: 1024 tokens (captures several progressions)
- **Positional Encoding**: EigenSpace chord reference as a positional dissonance perception model. 

#### Training Strategies

**Strategy A: Curriculum Learning** (Progressive)
1. Train on 12-TET only
2. Gradually introduce 10% microtonal
3. Progress to 50% microtonal
4. Finally train on 100% microtonal

**Strategy B: Mixed Training** (Recommended First Attempt)
- Train on all data levels simultaneously
- Let model learn the spectrum of microtonality naturally
- Simpler implementation

#### Evaluation Metrics

##### Quantitative
- Perplexity on held-out test set
- Token prediction accuracy

##### Qualitative (Musical)
- Generate progressions → render to MIDI → listen
- **Harmonic Function**: Does it preserve functional harmony?
- **Voice Leading**: Are transitions smooth and musically logical?
- **Microtonal Coherence**: Do 53-TET inflections follow EigenSpace principles?

**Status**: 🔄 Planned

---

## 🚀 Immediate Next Steps

1. **✅ XML → MIDI Pipeline** - Complete with validation (93.4% accuracy)
2. **✅ Voicing System** - 7 templates implemented and tested
3. **🔧 Run Full Transposition** - Execute 12-key augmentation on all 4,000 songs → 48,000
4. **⏳ Export MPE-MIDI Files** - Generate MPE format for all augmented progressions
5. **⏳ Implement Tokenizer** - Design metadata + MIDI token vocabulary
6. **⏳ Stage 3 Prototype** - Create microtonal substitution rules (10% level)
7. **⏳ Training Pipeline** - Prepare dataset for GPT-2 training

---

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
For more details, see the full license at [Creative Commons CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Or read the license document attached.