[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

# ANIMA Microtonal GPT

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





---

### **Stage 5: GPT-2 Model Training**

Train transformer model on hybrid 12-TET/53-TET sequences.

#### Architecture
- **Model Size**: GPT-2 Small (124M parameters) - sufficient for this domain
- **Context Window**: 4096 tokens (captures several progressions)
- **Positional Encoding**: EigenSpace chord reference as a positional dissonance perception model. 

#### Training Strategies


#### Evaluation Metrics

##### Quantitative
- Perplexity on held-out test set
- Token prediction accuracy

##### Qualitative (Musical)


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