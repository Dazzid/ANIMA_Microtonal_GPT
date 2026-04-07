from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm
from midiutil import MIDIFile

import formats as fmt
import librosa as lib
import utils_lenghts as lengths
import random
import math
import os

# some by default declarations
def getNotes():
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
    return notes

def getFormat():
    format = ['.', '<start>', '<end>', '<pad>']
    return format

#-------------------------------------------------------------------------
#Get the metadata and chords from the XML files 
def createCustomDataset(path, padding_length=512, onlyFourByFour=True):
    
    songFiles = []
    #first get only the .xml files to avoid hidden files as .DS_Store
    for item in os.listdir(path):
        if item.endswith('.xml') and os.path.isfile(os.path.join(path, item)):
            songFiles.append(item)

    #sort the songs in alphabetical order
    songFiles.sort()
    all_bass_notes = []
    all_durations = []
    all_chords = []
    all_relative_pos = []
    meta = []

    for item in tqdm(songFiles):
        song_path = path + '/' + item
        meta_info = get_metadata(song_path)
        if onlyFourByFour:  
            if (meta_info['time_signature'] == '4/4') or (meta_info['time_signature'] == '2/4'): 
                meta.append(meta_info)
                durations, relative_positions, chords, bass_notes = get_chords_from_file(song_path, False)
                #pad the arrays to the max length
                bass_notes = padding(bass_notes, padding_length)
                durations = padding(durations, padding_length)
                chords = padding(chords, padding_length)
                relative_positions = padding(relative_positions, padding_length)
                #append all arrays to the defined length
                all_bass_notes.append(bass_notes)
                all_durations.append(durations)
                all_chords.append(chords)
                all_relative_pos.append(relative_positions)
        else:
            meta.append(meta_info)
            durations, relative_positions, chords, bass_notes = get_chords_from_file(song_path)
            #pad the arrays to the max length
            bass_notes = padding(bass_notes, padding_length)
            durations = padding(durations, padding_length)
            chords = padding(chords, padding_length)
            relative_positions = padding(relative_positions, padding_length)
            #append all arrays to the defined length
            all_bass_notes.append(bass_notes)
            all_durations.append(durations)
            all_chords.append(chords)
            all_relative_pos.append(relative_positions)
            
    all_relative_pos = np.array(all_relative_pos, object)
    all_bass_notes = np.array(all_bass_notes)
    all_durations = np.array(all_durations)
    all_chords = np.array(all_chords)
    #meta = np.array(meta)
    return all_bass_notes, all_durations, all_chords, all_relative_pos, meta

#-------------------------------------------------------------------------
class TokenDatasetMidi(Dataset):
    def __init__(self, dataset, midi_dataset, block_size, tokens):
        self.dataset = dataset
        #print("midi shape:", midi_dataset.shape)
        self.midi_dataset = midi_dataset #n x L:512 x 8
        data_size, vocab_size = len(self.dataset ), len(tokens)
        print('data has %d pieces, %d unique tokens.' % (data_size, vocab_size))
        self.stoi = { tk:i for i,tk in enumerate(tokens) }
        self.itos = { i:tk for i,tk in enumerate(tokens) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx:idx+1]
        midi = self.midi_dataset[idx:idx+1][0] # 1 x 512 x 8
        
        #print(midi.shape)
        # encode every token to an integer
        dix = [self.stoi[s] for s in chunk[0]]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        m = torch.tensor(midi[:-1], dtype=torch.long)
        
        return x, y, m

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class TokenDatasetMidiEigen(Dataset):
    """Dataset that returns (x, y, m, e) — tokens, targets, midi, eigenspace coords."""
    def __init__(self, dataset, midi_dataset, eigen_dataset, block_size, tokens):
        self.dataset = dataset
        self.midi_dataset = midi_dataset      # n x L x 8
        self.eigen_dataset = eigen_dataset    # n x L x 4  (α, β, γ, D) per token
        data_size, vocab_size = len(self.dataset), len(tokens)
        print('data has %d pieces, %d unique tokens (with EigenSpace).' % (data_size, vocab_size))
        self.stoi = { tk:i for i,tk in enumerate(tokens) }
        self.itos = { i:tk for i,tk in enumerate(tokens) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx:idx+1]
        midi = self.midi_dataset[idx:idx+1][0]    # 512 x 8
        eigen = self.eigen_dataset[idx:idx+1][0]  # 512 x 4
        
        dix = [self.stoi[s] for s in chunk[0]]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        m = torch.tensor(midi[:-1], dtype=torch.long)
        e = torch.tensor(eigen[:-1], dtype=torch.float32)
        
        return x, y, m, e

#-------------------------------------------------------------------------class TokenDatasetMidiEigen(Dataset):
    """Dataset that returns (x, y, m, e) — token IDs, targets, MIDI, and EigenSpace coords."""
    def __init__(self, dataset, midi_dataset, eigen_dataset, block_size, tokens):
        self.dataset = dataset
        self.midi_dataset = midi_dataset    # n x L:512 x 8
        self.eigen_dataset = eigen_dataset  # n x L:512 x 4  (α, β, γ, D)
        data_size, vocab_size = len(self.dataset), len(tokens)
        print('data has %d pieces, %d unique tokens (with EigenSpace).' % (data_size, vocab_size))
        self.stoi = { tk:i for i,tk in enumerate(tokens) }
        self.itos = { i:tk for i,tk in enumerate(tokens) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx:idx+1]
        midi = self.midi_dataset[idx:idx+1][0]    # 512 x 8
        eigen = self.eigen_dataset[idx:idx+1][0]  # 512 x 4
        
        # encode every token to an integer
        dix = [self.stoi[s] for s in chunk[0]]
        
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        m = torch.tensor(midi[:-1], dtype=torch.long)
        e = torch.tensor(eigen[:-1], dtype=torch.float32)
        
        return x, y, m, e

#-------------------------------------------------------------------------class TokenDataset(Dataset):
    def __init__(self, dataset, block_size, tokens):
        self.dataset = dataset
        #print("midi shape:", midi_dataset.shape)
        data_size, vocab_size = len(self.dataset ), len(tokens)
        print('data has %d pieces, %d unique tokens.' % (data_size, vocab_size))
        self.stoi = { tk:i for i,tk in enumerate(tokens) }
        self.itos = { i:tk for i,tk in enumerate(tokens) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        chunk = self.dataset[idx:idx+1]
        # encode every token to an integer
        dix = [self.stoi[s] for s in chunk[0]]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        
        return x, y
    
#Dummy format ----------------------------------------------------------
def format_start_end(myData):
    myData = np.asarray([myData], object)
    myData = np.append(myData, '<end>')
    myData = np.insert(myData, 0, '<start>')
    return myData

#padding to make the length of the sequence equal to the block size
def padding(array, max_len):
        array = np.append(array,['<pad>']*(max_len-len(array) ))
        assert len(array) == max_len
        return np.array(array)
    
#get the data from the file
def get_project_root() -> Path:
    return Path(__file__).parent.parent

#extract the metadata from the file -------------------------------------
def get_metadata(path):
    metadata = {'composer': 'Null', 'style': 'Null', 'song_name': 'Null', 'tonality': 'Null', 'midi_key': 0, 'time_signature': '4/4', 'decade': 'Null'}
    tree = ET.parse(path)
    
    #get version of xml
    #print(ET.VERSION)
    
    root = tree.getroot()
   #print(root)
    #parse the name of the song
  
    title = 'None'
    
    # Attempt to find an element under the root
    title_element = root.find('work')
    
    if title_element != None:
        title = title_element.find('work-title').text        
    else:
        title = root.find('movement-title').text

    part = root.find('part')
    #define the metadata elements

    #released_year = 'Null'

    mode = 'major' # major, minor, 0, -1, etc

    #Extract metadata 
    info = root.find('identification').findall('creator')

    metadata['song_name'] = title
    metadata['composer'] = info[0].text if len(info) > 0 else 'Unknown'
    metadata['style'] = info[1].text if len(info) > 1 else 'Unknown'
    
    # Extract additional metadata from identification section
    identification = root.find('identification')
    if identification is not None:
        # Encoding info (software, date)
        encoding = identification.find('encoding')
        if encoding is not None:
            software = encoding.find('software')
            encoding_date = encoding.find('encoding-date')
            if software is not None:
                metadata['software'] = software.text
            if encoding_date is not None:
                metadata['encoding_date'] = encoding_date.text

    #print('Composer: ', metadata['composer'], '\nStyle: ', metadata['style'], '\nSong name: ',metadata['song_name'])

    #Extract the time signature
    total_notes_length = 0
    total_bars = 0 
    length_reference = 0.000325520834 #this is 1/3072 (samples for whole note)
    dict_fifth_cycle = lengths.getFifthCicle()
    for measure in part:
        #print('bar: ', measure.attrib['number'])
        total_bars = int(measure.attrib['number'])
        if(measure.attrib['number'] == '1'):
            data = measure.find('attributes')
            key = data.find('key').find('fifths').text
            mode = data.find('key').find('mode').text
            metadata['tonality'] = dict_fifth_cycle[mode][key]+ " " + mode
            #print('Tonality: ', metadata['tonality'])
            midi_key = lib.note_to_midi(dict_fifth_cycle[mode][key])
            metadata['midi_key'] = midi_key
            #print('Midi key tonality: ', metadata['midi_key'])
        #harmony_section = measure.findall('harmony')
        notes = measure.findall('note')
        notes_length_in_bar = 0
        for note in notes:
            duration = int(note.find('duration').text)
            duration_samples = length_reference * duration
            notes_length_in_bar += duration_samples
        
        total_notes_length += notes_length_in_bar
    divisor = '/4'
    time_signature = round(total_notes_length)*4 / total_bars
    if time_signature == 6:
        divisor = '/8'
    metadata['time_signature'] = str(int(time_signature)) + divisor
    #print('Time_signature:', metadata['time_signature'])
    return metadata

#Get the midi notes from the chords --------------------------------------------
def createMidiAndChord(chordList):
    midi_embeddings = get_the_midi_notes_chords(chordList)
    nature = getArrayOfElementsInChord(chordList)
    nature.pop() #remove the last element

    elements_per_chord=[]
    counter = 0
    for element in nature:
        if element != '.' and element != '<start>' and element != '<end>' and element != '<pad>':
            counter += 1
        if element == '.':
            elements_per_chord.append(counter)
            counter = 0

    elements_per_chord.remove(0)

    song_midi_embeddings = []
    x = 0
    ref = -1
    for element in nature:
        if element == '<start>' or element == '<end>' or element == '<pad>':
            song_midi_embeddings.append(np.zeros(8))
        if element == '.':
            x += 1
            ref += 1
            song_midi_embeddings.append(np.zeros(8))
        elif element != '<start>' and element != '<end>' and element != '<pad>':
            song_midi_embeddings.append(midi_embeddings[ref])

    song_midi_embeddings = np.array(song_midi_embeddings)
    return nature, song_midi_embeddings

#Counter of elements in chord ---------------------------------------------------
def counterOfElementsInChord(song):
    add = 0
    counter = [0, 0] #first two element are <style> and the actual style 
    for i in range(2, len(song)):
        e = song[i]
        if e == '<start>' or e == '<end>' or e == '<pad>' or e == '.':
            add = 0
            counter.append(add)
        else: 
            add += 1
            counter.append(add)
    return counter

#Correct the padding for midi --------------------------------------------------
def correctMidiEmbeddings(midi, data, theCounter):
    theCounter = theCounter + [0,0]

    for i in range(len(data)-1):
        note = data[i]
        #correct the base after the nature being shure it is not a slash
        if theCounter[i+1] == 2 and data[i+1] != '/':
            if note.find('-') != -1:
                note = note.replace('-', 'b')
            m = lib.note_to_midi(note) + 48
            midi[i] = [m] + [0,0,0,0,0,0,0]
        #correct the nature after the base before a slash
        if theCounter[i] == 2 and data[i] != '/' and data[i+1] == '/':
            symbol = data[i-1] + data[i]
            newChord = createChord(symbol)
            midi[i] = newChord
        #correct that slash is an special midi embedding
        if data[i] == '/':
            midi[i] = [0,0,0,0,0,0,0,127]
        #correct the base after the slash when it has nature
        if data[i] == '/' and theCounter[i+2] == 5:
            note = data[i+1]
            if note.find('-') != -1:
                note = note.replace('-', 'b')
            m = lib.note_to_midi(note) + 48
            midi[i+1] = [m] + [0,0,0,0,0,0,0]
        #correct the chord before the slash in the case there is no nature
        if data[i+1] == '/' and theCounter[i] == 1:
            symbol = data[i]
            newChord = createChord(symbol)
            midi[i] = newChord
        #correct the nature before an extension without slash
        if theCounter[i] == 1 and theCounter[i+1] == 2 and theCounter[i+2] == 3 and data[i+1] != '/' and data[i+2] != '/':
            symbol = data[i] + data[i+1]
            newChord = createChord(symbol)
            midi[i+1] = newChord
    
    last = data[-1]
    if (last) != '.' and len(last) <= 2 and last.isnumeric() == False:
        m = lib.note_to_midi(last) + 48
        midi[-1] = [m] + [0,0,0,0,0,0,0]
    if last == '.':
        midi.append([0,0,0,0,0,0,0,0])
    return midi

#Shuffle Dataset ----------------------------------------------------------------

#create a file with shuffled reference index
def createWindowedShuffleReference(size, window, save = False):
    s = np.arange(0, size, 1)
    #num = np.arange(0, len(data)/10, 1)
    np.random.shuffle(s)

    n = int(size/window)
    numlist = random.sample(range(n), n)
    numlist = np.array(numlist)
    numlist = numlist * window

    m = np.max(numlist)
    l_ref = size-window
    print('real:', size, 'max:', m, 'length_ref:',l_ref)

    if m != l_ref:
        rest = m - l_ref
        numlist = numlist - rest

    ref = []
    for num in numlist:
        if num == 0:
            print("OK")
        for i in range(0,window):
            ref.append(num+i)

    #return the shuffled list
    if save:
        np.savetxt("../dataset/shuffle_order.txt", ref, fmt='%i', delimiter=" ", header='Array shape: ('+str(size)+', 1)')
    return ref

#Data Split ----------------------------------------------------------------
def generateDatasetSplit(db, split=0.1):
    num = int(len(db)*split)
    print(num)
    if (num %2) != 0:
        num += 1 
    training, test = db[num:,:], db[:num,:] 
    return training, test 

#Save sessions ----------------------------------------------------------------
import copy
import json
MODEL_NAME = "session_model"

def save_metadata_to_json(metadata, output_dir="../dataset/metadata"):
    """
    Save metadata dictionary as individual JSON file per song.
    
    Args:
        metadata: Dictionary with song metadata
        output_dir: Directory to save JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean filename (remove invalid characters)
    song_name = metadata.get('song_name', 'Unknown')
    safe_filename = song_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    json_path = os.path.join(output_dir, f"{safe_filename}.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return json_path

def export_all_metadata_from_xml(xml_dir="../dataset/iRealXML", output_dir="../dataset/metadata"):
    """
    Export metadata from all XML files as individual JSON files.
    
    Args:
        xml_dir: Directory containing XML files
        output_dir: Directory to save JSON files
        
    Returns:
        List of exported JSON file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = []
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    print(f"Exporting metadata for {len(xml_files)} songs...")
    
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        
        try:
            metadata = get_metadata(xml_path)
            json_path = save_metadata_to_json(metadata, output_dir)
            exported_files.append(json_path)
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    print(f"Exported {len(exported_files)} metadata files to {output_dir}")
    return exported_files

def save_model(MODEL_NAME, model):
    # SAVE THE SESSION MODEL 
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), MODEL_NAME)
    print('Model', MODEL_NAME, 'saved')
    
def load_model(MODEL_NAME, model):
    ckpt_model = model.module if hasattr(model, "module") else model
    try:
        ck = torch.load(MODEL_NAME)
    except:
        return None
    ckpt_model.load_state_dict(copy.deepcopy(ck))
    model.cuda()
    print('Checkpoint loaded', MODEL_NAME)
    return model

#-------------------------------------------------------------------------
# MPE MIDI Export Class
#-------------------------------------------------------------------------
class MPE_MIDI_Exporter:
    """
    Export MIDI files with MPE (MIDI Polyphonic Expression) support.
    MPE allows independent pitch bend per note, essential for microtonal music.
    """
    
    def __init__(self, num_channels=15, pitch_bend_range=48):
        """
        Initialize MPE MIDI Exporter
        
        Args:
            num_channels: Number of note channels (2-16), channel 1 is master
            pitch_bend_range: Pitch bend range in semitones (default ±48 for microtonal)
        """
        self.num_channels = num_channels
        self.pitch_bend_range = pitch_bend_range
        self.master_channel = 0  # Channel 1 (0-indexed)
        self.note_channels = list(range(1, num_channels + 1))  # Channels 2-16
        self.current_channel_idx = 0
        
    def get_next_channel(self):
        """Round-robin channel allocation for polyphony"""
        channel = self.note_channels[self.current_channel_idx]
        self.current_channel_idx = (self.current_channel_idx + 1) % len(self.note_channels)
        return channel
    
    def setup_mpe_channels(self, midi_file):
        """
        Configure MPE channels with proper settings
        - Set pitch bend range for all channels
        - Configure master channel
        """
        # Set pitch bend range on master channel (RPN MSB/LSB for pitch bend sensitivity)
        for channel in [self.master_channel] + self.note_channels:
            # RPN for pitch bend sensitivity
            midi_file.makeRPNCall(track=0, channel=channel, time=0, 
                                 controller_msb=0, controller_lsb=0, 
                                 data_msb=self.pitch_bend_range, data_lsb=0)
    
    def export_to_mpe_midi(self, midi_voicing_data, filename, output_path='./'):
        """
        Export MIDI voicing data to MPE-formatted MIDI file.
        
        SIMPLE APPROACH: The sequence from convert_chords_to_voicing() now contains
        ONLY chords (one per DOT). Each element is (midi_array, duration, label).
        Just export all of them in order - no complex DOT-search logic needed!
        
        Args:
            midi_voicing_data: List of 3-tuples [(midi_notes_array, duration, chord_name), ...]
            filename: Output filename
            output_path: Directory to save file
        """
        import random
        
        # Create MIDI file with 1 track
        midi = MIDIFile(1, adjust_origin=False)
        track = 0
        tempo = 120  # BPM
        
        midi.addTempo(track, 0, tempo)
        
        # Setup MPE channels
        self.setup_mpe_channels(midi)
        
        # Export all chords directly - no complex filtering needed!
        current_time = 0.0
        
        for item in midi_voicing_data:
            midi_notes = item[0]
            duration = float(item[1])
            
            # Get non-zero notes
            active_notes = [note for note in midi_notes if note > 0]
            
            # Skip if no notes
            if len(active_notes) == 0:
                continue
            
            # Convert duration from seconds to beats
            # At 120 BPM: 1 second = 2 beats
            duration_in_beats = duration * (tempo / 60.0)
            
            # Add each note to a separate MPE channel
            for note in active_notes:
                channel = self.get_next_channel()
                velocity = int(random.uniform(55, 85))
                
                # Add note on MPE channel
                midi.addNote(track=track, 
                           channel=channel, 
                           pitch=int(note), 
                           time=current_time, 
                           duration=duration_in_beats, 
                           volume=velocity)
            
            # Advance time
            current_time += duration_in_beats
        
        # Write MIDI file
        full_path = f"{output_path}/{filename}.mid"
        with open(full_path, 'wb') as output_file:
            midi.writeFile(output_file)
        
        return full_path
