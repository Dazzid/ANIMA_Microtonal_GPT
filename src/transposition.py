import formats as fmt


class Transposition:
    #define the class
    def __init__(self):
        self.chromatic_scale = [
            ('C',), ('C#', 'Db'), ('D',), ('D#', 'Eb'), ('E'), ('F',), ('F#', 'Gb'),
            ('G',), ('G#', 'Ab'), ('A',), ('A#', 'Bb'), ('B')
        ]
        self.line_of_fifths = [
           'Fbb', 'Cbb', 'Gbb', 'Dbb', 'Abb', 'Ebb', 'Bbb', 'Fb', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C','G',
           'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#', 'F##', 'C##', 'G##', 'D##', 'A##', 'E##', 'B##'
        ]
        
        #Durations 
        self.length_durations = {'0.3997395833333333', '0.4440104166666667', '0.5', '0.5703125',
                '0.6666666666666666', '0.75', '0.7994791666666666', '0.8880208333333334',
                '1.0', '1.1419270833333333', '1.3333333333333333', '1.5',
                '1.5989583333333333', '1.7135416666666667', '2.0', '2.25',
                '2.3997395833333335', '2.6666666666666665', '3.0', '4.0'}
        
        self.notes_in_the_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        # Enharmonic equivalents and mappings updated to include minor keys properly
        self.enharmonic_equivalents = {
            'C#': 'Db',
            'Db': 'C#',
            'D#': 'Eb',
            'Eb': 'D#',
            'F#': 'Gb',
            'Gb': 'F#',
            'G#': 'Ab',
            'Ab': 'G#',
            'A#': 'Bb',
            'Bb': 'A#',
            'E#': 'F',
            'B#': 'C',
            'Cb': 'B',
            'B': 'Cb',
            'E': 'Fb',
            'Fb': 'E'
        }

        self.major_keys = {
            'C':    [],
            'G':    ['F#'],
            'D':    ['F#', 'C#'],
            'A':    ['F#', 'C#', 'G#'],
            'E':    ['F#', 'C#', 'G#', 'D#'],
            'B':    ['F#', 'C#', 'G#', 'D#', 'A#'],
            'F#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#'],
            'C#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#'],
            'F':    ['Bb'],
            'Bb':   ['Bb', 'Eb'],
            'Eb':   ['Bb', 'Eb', 'Ab'],
            'Ab':   ['Bb', 'Eb', 'Ab', 'Db'],
            'Db':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb'],
            'Gb':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'],
            'Cb':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb']
        }

        self.minor_keys = {
            # Adding the correct mappings for minor keys based on their relative majors
            'A':    [],
            'E':    ['F#'],
            'B':    ['F#', 'C#'],
            'F#':   ['F#', 'C#', 'G#'],
            'C#':   ['F#', 'C#', 'G#', 'D#'],
            'G#':   ['F#', 'C#', 'G#', 'D#', 'A#'],
            'D#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#'],
            'A#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#'],
            'D':    ['Bb'],
            'G':    ['Bb', 'Eb'],
            'C':    ['Bb', 'Eb', 'Ab'],
            'F':    ['Bb', 'Eb', 'Ab', 'Db'],
            'Bb':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb'],
            'Eb':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'],
            'Ab':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb']
        }
        
        self.dominant_keys = {
            'G':    [],
            'D':    ['F#'],
            'A':    ['F#', 'C#'],
            'E':    ['F#', 'C#', 'G#'],
            'B':    ['F#', 'C#', 'G#', 'D#'],
            'F#':   ['F#', 'C#', 'G#', 'D#', 'A#'],
            'C#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#'],
            'G#':   ['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#'],
            'C':    ['Bb'],
            'F':    ['Bb', 'Eb'],
            'Bb':   ['Bb', 'Eb', 'Ab'],
            'Eb':   ['Bb', 'Eb', 'Ab', 'Db'],
            'Ab':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb'],
            'Db':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'],
            'Gb':   ['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb']
        }
    
    #------------------------------------------------------------------------
    # Function to generate a dictionary for a given starting note
    def generate_chromatic_dict(self, start_note):
        # Find the start index
        start_index = next(i for i, notes in enumerate(self.chromatic_scale) if start_note in notes)
        
        # Generate the dictionary
        note_dict = {}
        for i in range(12):
            # Calculate the current index
            current_index = (start_index + i) % 12
            # Get the note(s) at the current index
            current_notes = self.chromatic_scale[current_index]
            
            # For each enharmonic equivalent at this index, set the distance
            for note in current_notes:
                note_dict[note] = i
        
        return note_dict
    
    #------------------------------------------------------------------------
    # Function to get the chromatic scale for a given note
    def getChromaticScale(self, note):
        return self.generate_chromatic_dict(note)   
         
    #------------------------------------------------------------------------
    # Function to get the distance between two notes
    def get_distance_line_of_fifth(self, previous, current):    
        distance = self.line_of_fifths.index(current) - self.line_of_fifths.index(previous)
        return distance
    
    #------------------------------------------------------------------------
    # Function to get the note from the line of fifths
    def get_note_from_steps_distance(self, note, steps):
        index = self.line_of_fifths.index(note)
        new_note = self.line_of_fifths[(index + steps) % len(self.line_of_fifths)]
        return new_note
    
    #------------------------------------------------------------------------
    # Function to transpose a note by a given interval
    def get_alterations_scales(self, tonality):
        if '##' in tonality:
            this_note = tonality.split(' ')[0]
            this_mode = tonality.split(' ')[1]
            this_idx = self.line_of_fifths.index(this_note)
            new_note = self.line_of_fifths[this_idx-12]
            tonality = new_note + ' ' + this_mode
            
        elif 'bb' in tonality:
            this_note = tonality.split(' ')[0]
            this_mode = tonality.split(' ')[1]
            this_idx = self.line_of_fifths.index(this_note)
            new_note = self.line_of_fifths[this_idx+12]
            tonality = new_note + ' ' + this_mode
        
        mode = 'None'
        if 'major' in tonality:
            mode = 'major'
        elif 'minor' in tonality:
            mode = 'minor'
        elif 'dominant' in tonality:
            mode = 'dominant'
        
        root_note, quality = tonality.split(' ')
        alterations = None
        enharmonic_alterations = None
        
        if mode == 'minor':
            alterations = self.minor_keys.get(root_note)
            enharmonic_alterations = self.minor_keys.get(self.enharmonic_equivalents.get(root_note))
        elif mode == 'dominant':
            alterations = self.dominant_keys.get(root_note)
            enharmonic_alterations = self.dominant_keys.get(self.enharmonic_equivalents.get(root_note))
        elif mode == 'major':
            alterations = self.major_keys.get(root_note)
            enharmonic_alterations = self.major_keys.get(self.enharmonic_equivalents.get(root_note))
            
        #print(alterations, enharmonic_alterations)
        
        if enharmonic_alterations != None and alterations != None:
            if len(alterations) > len(enharmonic_alterations):
                alterations = enharmonic_alterations
                root_note = self.enharmonic_equivalents.get(root_note)
                #print(f"Enharmonic equivalent found: {root_note}")
                tonality = f"{root_note} {quality}"
            
        corrected_tonality = tonality
        
        if alterations == None:
            enharmonic_root = self.enharmonic_equivalents.get(root_note)
            #print(f"Enharmonic equivalent found: {enharmonic_root}")
            if enharmonic_root:
                corrected_tonality = f"{enharmonic_root} {quality}"
                if mode == 'minor':
                    alterations = self.minor_keys.get(enharmonic_root, "Unknown key")
                elif mode == 'dominant':
                    alterations = self.dominant_keys.get(enharmonic_root, "Unknown key")
                elif mode == 'major':
                    alterations = self.major_keys.get(enharmonic_root, "Unknown key")
            else:
                alterations = "Unknown key"
                print("Error, unknown key", tonality, mode)
        
        scales = []
        
        starting_note = corrected_tonality.split(' ')[0]
        if len(starting_note) > 1:
            note = starting_note[:1]
        else:
            note = starting_note
        
        idx = self.notes_in_the_scale.index(note)
        #print(idx)
        while len(scales) < 7:
            thisNote = self.notes_in_the_scale[idx]
            if thisNote + 'b' in alterations:
                thisNote = thisNote + 'b'
            elif thisNote + '#' in alterations:
                thisNote = thisNote + '#'
            scales.append(thisNote)
            idx = idx+ 1
            idx = idx % 7    
        
        return corrected_tonality, alterations, scales
    
    #------------------------------------------------------------------------
    # Function to find notes corresponding to a specific number (semitone position)
    def find_notes_for_semitone(self, semitone_position):
        note_to_semitones = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        return [note for note, position in note_to_semitones.items() if position == semitone_position]

    #------------------------------------------------------------------------
    # Function to calculate the distance between two notes
    def get_note_distance(self, from_note, to_note):
        # Lists for natural and sharp notes, and a separate one for flats
        sharp_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        flat_notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        
        to_index = -1
        from_index = -1
        
        # Attempt to calculate distance using sharp notes first
        if from_note in sharp_notes:
            from_index = sharp_notes.index(from_note)
        elif from_note in flat_notes:
            from_index = flat_notes.index(from_note)    
            
        if to_note in sharp_notes:
            to_index = sharp_notes.index(to_note)
        elif to_note in flat_notes:
            to_index = flat_notes.index(to_note)
        
        if from_index == -1 or to_index == -1:
            print('Error: Check the input notes', from_note, to_note)
            return -1
        distance = (to_index - from_index) % 12

        return distance
    
    #------------------------------------------------------------------------
    # Function to calculate the distance between two notes
    def get_octatonic_distance(self, tonality, noteReference):
        _,_,myScale = self.get_alterations_scales(tonality)
        #print(myScale)
        if noteReference in myScale:
            distance = myScale.index(noteReference)
        elif noteReference + 'b' in myScale:
            distance = myScale.index(noteReference + 'b')
        elif noteReference + '#' in myScale:
            distance = myScale.index(noteReference + '#')
        elif len(noteReference) > 1 and noteReference[:1] in myScale:
            distance = myScale.index(noteReference[:1])
        else:
            distance = -1
        return distance
    
    #--------------------------------------------------------------------------------
    #Transpose the sequence
    def transpose_song(self, sequence, tonality, new_tonality):
        '''
        Get the sequence, extract the song information, the chords location and its duration and 
        transpose to a new tonality.
        '''
        #print('from:', tonality, 'to:', new_tonality)
        tonality_note = tonality.split(' ')[0]

        #Extract the notes
        check_notes = fmt.getNotes()
        
        song_to_translate_notes = []

        other_elements = fmt.listToIgnore()
        # Extract chords information with its location and duration
        
        for i, element in enumerate(sequence):
            info = {'note': '', 'distance': -1, 'nature': 'Null', 'loc': -1, 'duration': -1}
            #element = item[0]
            #duration = item[1]
            if element in self.length_durations:
                duration = element
            if element in check_notes:
                info['note'] = element
                info['loc'] = i
                info['duration'] = duration
                if i < len(sequence)-1: this_nature = sequence[i+1]
              
                if this_nature in other_elements:
                    this_nature = 'maj'
                info['nature'] = this_nature
                song_to_translate_notes.append(info)     
            
        for i, item in enumerate(song_to_translate_notes):
            current_note = item['note']
            if i == 0:
                previous_note = tonality_note
            else:
                previous_note = song_to_translate_notes[i-1]['note']

            distance = self.get_distance_line_of_fifth(previous_note, current_note)
        
            song_to_translate_notes[i]['distance'] = distance
            
        new_note = ''
        transposed = ''
        transposed_song = []
        
        #Transpose the notes based in the line of fifths
        for i, item in enumerate(song_to_translate_notes):
            new_item = item.copy()
            current_note = item['note']
            if i == 0:
                previous_note = new_tonality
            else:
                previous_note = new_note
                
            steps = item['distance']
            new_note = self.get_note_from_steps_distance(previous_note, steps)
            #check its validity 
            
            nature = item['nature'][0]
            check_note = None
            transposed = new_note
            
            if nature == 'maj' or nature == 'maj7' or nature == 'maj6':
                mode = 'major'
                check_note, _, _ = self.get_alterations_scales(transposed+' '+mode)
                transposed = check_note.split(' ')[0]
                
            elif nature == 'm' or nature == 'm7' or nature == 'm6':
                mode = 'minor'
                check_note, _, _ = self.get_alterations_scales(transposed+' '+mode)
                transposed = check_note.split(' ')[0]
                
            elif nature == 'dom7':
                mode = 'dominant'
                check_note, _, _ = self.get_alterations_scales(transposed+' '+mode)
                transposed = check_note.split(' ')[0]
            
            elif check_note != None:
                transposed = check_note.split(' ')[0]
            
            new_item['note'] = transposed
            transposed_song.append(new_item)
        
        #Correct the fifths enharmonics
        for i, item in enumerate(transposed_song):
            if i < len(transposed_song) - 1:
                next_note = transposed_song[i+1]['note']
                next_step = transposed_song[i+1]['distance']
                current_note = item['note']
                if next_step == -1:
                    distance = self.get_distance_line_of_fifth(current_note, next_note)
                    #print(current_note, next_note, distance)
                    if distance == -13:
                        assert True, "Distance chord not correct"
                        new_note = self.get_note_from_steps_distance(next_note, 1)
                        result ,_ ,_ = self.get_alterations_scales(new_note+' major')
                        new_note = result.split(' ')[0]
                        transposed_song[i]['note'] = new_note
        
        #create a new file 
        transposed_final = sequence.copy()
        for i, element in enumerate(transposed_song):
            transposed_final[element['loc']] = element['note']
        return transposed_final
            
         
