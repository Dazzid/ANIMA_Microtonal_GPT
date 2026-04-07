import numpy as np
import librosa as lib
from tqdm.auto import tqdm

# some by default declarations
def getNotes():
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb']
    return notes

def getNatures():
    natures = ['maj', 'maj6', 'maj7', 'm', 'm7', 'm_maj7', 'm6', 'dom7', 'ø7', 'o7', 'o', 'power', 'sus', 'sus7', 'sus2', 'sus4', 'aug', 'o_maj7']
    return natures

def getFormat():
    format = ['.', '<start>', '<end>', '<pad>']
    return format

def listToIgnore():
    ignore_list = {'<start>', '<end>', '<pad>', '.', '|', '||', 'b||', 'e||', 'Repeat_0', 'Repeat_1', 'Repeat_2', 'Repeat_3', 'Intro', 
                    'Form_A', 'Form_B', 'Form_C', 'Form_D', 
                    'Form_verse', 'Form_intro', 'Form_Coda', 'Form_Head', 
                    'Form_Segno', '|:', ':|'}
    return ignore_list

#----------------------------------------------------------------------
def splitChordTokens(chord):
    '''This function splits the chord tokens into a list of tokens
    input is a string of the chord
    output is a list of tokens
    '''
    list = []  
    section = chord.split(' ')
    base = section[0][:1]
    nature = section[0][1:]
                      
    #fix the shifted error when sharp and flat appears   
         
    if '###' in nature or 'bbb' in nature:
        base = section[0][:4]
        nature = section[0][4:]
    
    elif '##' in nature or 'bb' in nature:
        base = section[0][:3]
        nature = section[0][3:]
    
    elif '#' in nature or 'b' in nature:
        base = section[0][:2]
        nature = section[0][2:]
                
    ext = ''
    for i in range(1, len(section)):
        ext += section[i]+ ' '
    ext = ext[:-1]
    
    list.append(base)
    list.append(nature)
    
    #add = ext.count('add')
    #subtract = ext.count('subtract')
    #alter = ext.count('alter')
    #print(add, subtract, alter)
    
    #check if the chord has extensions
    subSection = ext.split(' ')
  
    if (len(subSection) >  0):
        #print(subSection)
        line = ''
        for i in range(len(subSection)):
            line += subSection[i] + ' '
            #print(line)
        if line[-1] == ' ':
            line = line.rstrip()
        list.append(line)
    
    discard = ''
    while discard in list: list.remove(discard)
    return list

#------------------------------------------------------------------------------
def splitSlashChords(chord):
    '''This function splits the slash chords into a list of tokens
    input is a string of the chord
    output is a list of tokens
    '''
    list = []
    section = chord.split('/')
    list = splitChordTokens(section[0]) + ['/'] + splitChordTokens(section[1])
    return list

#------------------------------------------------------------------------------
#Split the chords
def splitTheChords(chords):
    ignore_list = listToIgnore()
    chord_list = []
    
    for chord in chords:
        if chord in ignore_list:
            chord_list.append(chord)
            continue
        
        chord_list.append('.')
        
        if '/' in chord:
            result = splitSlashChords(chord)
        else :
            result = splitChordTokens(chord)
            
        chord_list.extend(result)
        
    return chord_list

#------------------------------------------------------------------------------
#Get the array of elements per chord and also configure the offsets
def getArrayOfElementsInChord(chords, offsets):
    ignore_list = listToIgnore()
    chord_list = []
    offset_list= []
    
    #print(type(chords))
    for chord, offset in zip(chords, offsets):
        if chord in ignore_list:
            chord_list.append(chord)
            offset_list.append(offset)
            continue
        
        chord_list.append('.')
        offset_list.append(offset)
        
        chord_list.append(str(offset))
        offset_list.append(offset)
        
        if '/' in chord:
            result = splitSlashChords(chord)
        else :
            result = splitChordTokens(chord)
            
        chord_list.extend(result)
        offset_list.extend([offset] * len(result))
    return chord_list, offset_list


#-------------------------------------------------------------------------
#Correct the format of the chords
def correctTheFormat(reference, toBeCorrected):
    '''
    This function will correct the format of the chords
    '''
    new_dataset = []
    for origin, to_correct in zip(reference, toBeCorrected):
        #to_correct = transposed_data[0]
        to_correct = [s.replace('-', 'b') for s in to_correct]       
        to_correct = splitTheChords(to_correct)

        chord_id = 0
        fixed = []
        local_format = origin[:3]

        for i in range(3, len(origin)):
            element = origin[i]
            check = listToIgnore()
            check.remove('.') #dot is part of the chord
            
            if element in check: #this is asking if it is a chord
                fixed.append(element)
            else:
                try:
                    newChord = to_correct[chord_id]
            
                    #print(element, chord_id, newChord)
                    fixed.append(newChord)
                    chord_id += 1
                except:
                    print('error', i, '--->', element, chord_id, newChord)
                    break
                
        #add the format back
        fixed = list(local_format) + fixed
        new_dataset.append(fixed)
    
    new_dataset = np.array(new_dataset, dtype=object)
    return new_dataset

#------------------------------------------------------------------------------
#Correct style tokens by reference
def correctStyleTokensInMeta(sequence):
    #Get all the elements in meta and correct the style tokens
    for x, item in enumerate(tqdm(sequence)):
        element = item['style']
        if element == 'Moderately':
            sequence[x]['style'] = 'Pop'
        if element == "Even 8th's" or element == "Even 8's":
            sequence[x]['style'] = 'Even 8ths'
        if (element.find('Swing') != -1):
            sequence[x]['style'] = 'Jazz'
        if (element.find('Blues') != -1):
            sequence[x]['style'] = 'Blues'
        if (element.find('Folk') != -1):
            sequence[x]['style'] = 'Folk'
        if (element.find('Fusion') != -1):
            sequence[x]['style'] = 'Jazz'
        if (element.find('Jazz') != -1):
            sequence[x]['style'] = 'Jazz'
        if (element.find('Bossa') != -1):
            sequence[x]['style'] = 'Bossa'
        if (element.find('Reggae') != -1):
            sequence[x]['style'] = 'Reggae'
        if (element.find('Folk') != -1):
            sequence[x]['style'] = 'Folk'
        if (element.find('Samba') != -1):
            sequence[x]['style'] = 'Samba'
        if (element.find('Funk') != -1):
            sequence[x]['style'] = 'Funk'
        if (element.find('Pop') != -1):
            sequence[x]['style'] = 'Pop'
        if (element.find('Son') != -1):
            sequence[x]['style'] = 'Son'
        if (element.find('Rock') != -1):
            sequence[x]['style'] = 'Rock'
        if (element.find('Soul') != -1):
            sequence[x]['style'] = 'Soul'
        if (element.find('Balad') != -1):
            sequence[x]['style'] = 'Balad'
        if element == "R'n'B":
            sequence[x]['style'] = "RnB"
        if element == 'Beatles': 
            sequence[x]['style'] = 'Rock'
        if element == 'Afoxe': 
            sequence[x]['style'] = 'Afoxé'
        if element == 'Worship': 
            sequence[x]['style'] = 'Gospel'
        if element == 'Traditional Gospel': 
            sequence[x]['style'] = 'Gospel'
        if element == 'Deliberately': 
            sequence[x]['style'] = 'Pop'
    
#------------------------------------------------------------------------------
#Correct style tokens by reference in meta data
def correctStyleTokens(data):
    for x, song in enumerate(tqdm(data)):
        element = song[1]
        if element == 'Moderately':
            song[1] = 'Pop'
        if element == "Even 8th's" or element == "Even 8's":
            song[1] = 'Even 8ths'
        if (element.find('Swing') != -1):
            song[1] = 'Jazz'
        if (element.find('Blues') != -1):
            song[1] = 'Blues'
        if (element.find('Folk') != -1):
            song[1] = 'Folk'
        if (element.find('Jazz') != -1):
            song[1] = 'Jazz'
        if (element.find('Fusion') != -1):
            song[1] = 'Jazz'
        if (element.find('Bossa') != -1):
            song[1] = 'Bossa'
        if (element.find('Reggae') != -1):
            song[1] = 'Reggae'
        if (element.find('Folk') != -1):
            song[1] = 'Folk'
        if (element.find('Samba') != -1):
            song[1] = 'Samba'
        if (element.find('Funk') != -1):
            song[1] = 'Funk'
        if (element.find('Pop') != -1):
            song[1] = 'Pop'
        if (element.find('Son') != -1):
            song[1] = 'Son'
        if (element.find('Rock') != -1):
            song[1] = 'Rock'
        if (element.find('Soul') != -1):
            song[1] = 'Soul'
        if (element.find('Balad') != -1):
            song[1] = 'Balad'
        if element == "R'n'B":
            song[1] = "RnB"
        if element == 'Beatles': 
            song[1] = 'Rock'
        if element == 'Afoxe': 
            song[1] = 'Afoxé'
        if element == 'Worship': 
            song[1] = 'Gospel'
        if element == 'Traditional Gospel': 
            song[1] = 'Gospel'
        if element == 'Deliberately': 
            song[1] = 'Pop'
                
