import sys
sys.path.append(r'C:\Users\Creative\Documents\GitHub\music21')
from music21 import*
import numpy as np
import json
import glob
import os
import pickle
from midiutil.MidiFile import MIDIFile

def saveobj(dict,filename):
    '''
    Save object to disk
    '''
    try:
        outfile = open(filename,'wb')
        pickle.dump(dict,outfile)
        outfile.close()
    except Exception as e:
        print(e)

def loadobj(filename):
    '''
    Load object from memory
    '''
    try:
        infile = open(filename,'rb')
        new_dict =pickle.load(infile)
        infile.close()
        return new_dict
    except Exception as e:
        print(e)
        return None

def getmidimusic(path):
    '''
    Get other music that are in midi format(outside of Music21's corpus)
    '''
    p_dict = {}
    d_dict = {}
    encoder_dict = {}
    dataset = []
    for filename in glob.glob(os.path.join(path, '*.midi')):
        try:
            stm = converter.parse(filename)
            song = []
            for a_chord in stm.chordify():
                try:
                    chord = []
                    for a_note in a_chord:
                        if p_dict.__contains__(a_note.name)==False:
                            p_dict.__setitem__(a_note.name,len(p_dict))

                        if d_dict.__contains__(str(a_note.duration.quarterLength))==False:
                            d_dict.__setitem__(str(a_note.duration.quarterLength),len(d_dict))

                        if encoder_dict.__contains__(str(p_dict[a_note.name])+';'+str(d_dict[str(a_note.duration.quarterLength)]))==False:
                            encoder_dict.__setitem__(str(p_dict[a_note.name])+';'+str(d_dict[str(a_note.duration.quarterLength)]),len(encoder_dict))

                        chord.append(encoder_dict[str(p_dict[a_note.name])+';'+str(d_dict[str(a_note.duration.quarterLength)])])
                    song.append(chord)
                except Exception as e:
                    print(e)
                    pass
            dataset.append(song)
        except Exception as e:
            print(e)
            pass
    saveobj(p_dict,'pitch')
    saveobj(d_dict,'duration')
    saveobj(encoder_dict,'encoding')
    saveobj(dataset,'pop_dataset')

def get_melody(dataset):
    '''
    Extracts only the melody from a song, that is, the first part of the piece
    '''
    melody_dataset = []
    for song in dataset:
        temp = []
        for chord in song:
            temp.append(chord[0])
        melody_dataset.append(temp)
    return melody_dataset

class music(object):
    def __init__(self,n=1):
        self.data = getRawData(n)

        self.chord_data = []

        #self.data[0].show()
        #Used to represent musical features numerical
        self.pitch_dict ={}
        self.duration_dict = {}
        self.octave_dict ={}
        self.encoding_dict = {}

        self.song_length = 1000
        for i in range(len(self.data)):
            self.data[i] = get_chords(standerdizeMusic(self.data[i]))
            if len(self.data[i]) < self.song_length:
                self.song_length = len(self.data[i])

        self.pitch_oct = {}

        self.populate_dict()

        #Store dictionaries and data
        saveobj(self.data,'./Files/BachData')
        saveobj(self.pitch_dict,'./Files/BachPitch')
        saveobj(self.duration_dict,'./Files/BachDuration')
        saveobj(self.octave_dict,'./Files/BachOctaves')
        saveobj(self.encoding_dict,'./Files/BachEncoding')
        saveobj(self.chord_data,'./Files/BachChords')
        saveobj(self.pitch_oct,'./Files/BachPitchOctave')


    '''
    Populate dictionaries to switch to and from numerical represenation of a musical note
    to a music21 Note obj.
    '''
    def populate_dict(self):
        #Assign a number to each unique feature
        for song_index in range(len(self.data)):
            temp_song = []
            for chord_index in range(len(self.data[song_index])):
                temp_chord = []
                for note_index in range(4): #single note of a chord, 4 notes (parts)
                    if self.pitch_dict.__contains__(self.data[song_index][chord_index][note_index].name)==False:
                        self.pitch_dict.__setitem__(self.data[song_index][chord_index][note_index].name,len(self.pitch_dict))

                    if note_index==0 and self.duration_dict.__contains__(self.data[song_index][chord_index][note_index].duration.quarterLength)==False:
                        self.duration_dict.__setitem__(self.data[song_index][chord_index][note_index].duration.quarterLength,len(self.duration_dict))

                    if self.octave_dict.__contains__(self.data[song_index][chord_index][note_index].octave)==False:
                        self.octave_dict.__setitem__(self.data[song_index][chord_index][note_index].octave,len(self.octave_dict))

                    if self.encoding_dict.__contains__(str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.duration_dict[self.data[song_index][chord_index][note_index].duration.quarterLength]))==False:
                        self.encoding_dict.__setitem__(str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.duration_dict[self.data[song_index][chord_index][note_index].duration.quarterLength]),len(self.encoding_dict))

                    if self.pitch_oct.__contains__(str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.octave_dict[self.data[song_index][chord_index][note_index].octave]))==False:
                        self.pitch_oct.__setitem__(str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.octave_dict[self.data[song_index][chord_index][note_index].octave]),len(self.pitch_oct))

                    temp_chord.append(self.pitch_oct[str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.octave_dict[self.data[song_index][chord_index][note_index].octave])]) #Add pitch with its octave

                    #Convert Note obj to numerical format
                    #self.data[song_index][chord_index][note_index] = self.encoding_dict[str(self.pitch_dict[self.data[song_index][chord_index][note_index].name])+';'+str(self.duration_dict[self.data[song_index][chord_index][note_index].duration.quarterLength])]
                temp_chord.append(self.duration_dict[self.data[song_index][chord_index][0].duration.quarterLength])#time is duration of first note
                temp_song.append(temp_chord)
            self.chord_data.append(temp_song)

def getRawData(amount):
    '''
    Return stream objects of music(Bach corpus from music21)
     - 4-part chorales
     - 4 beats per bar
    '''
    print("---Retrieving music---")
    d = list()
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='filename'):
        d.append(score)
    #353 BACH CHORALES print(len(d))
    if amount==1:
        print(d[np.random.randint(0,len(d)-1)])
        return corpus.parse(d[np.random.randint(0,len(d)-1)])

    count = 0
    dataset = list()
    while count < amount:
        s = corpus.parse(d.pop())
        if len(s.parts)==4:
            if s.parts[0].getTimeSignatures().timeSignature.numerator==4 and s.parts[0].getTimeSignatures().timeSignature.denominator==4:
                dataset.append(s)
                count += 1
    return dataset

def standerdizeMusic(score):
    '''
    Transpose music to A-minor or C-major
    Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
    '''
    # conversion tables: e.g. Ab -> C is up 4 semitones, D -> A is down 5 semitones
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("C#",-1), ("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("F#",6), ("G-", 6), ("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("C#",-4),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#",3),("G-",3),("G", 2)])

    # transpose score
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    tScore = score.transpose(halfSteps)

    # transpose key signature
    for ks in tScore.flat.getKeySignatures():
        ks.transpose(halfSteps, inPlace=True)
    return tScore

def get_chords(song):
    '''
    Returns an array with each chord
    '''
    chords = []
    for some_measure in song.chordify():
        try:
            for some_chord in some_measure.elements:
                chord  = []
                for some_note in some_chord:
                    if getattr(some_note,'isNote',None):
                        chord.append(some_note)
                while len(chord)<4:
                    chord.append(chord[0])
                chords.append(chord)
        except Exception as e:
            print("WARNING(get_chords): ",e)
    return chords

def create_music(note_list):
    '''
    Concatenate notes into a music21 stream obj and save piece as midi file
    '''
    print("---Converting to MXL and Saving---")
    import datetime
    fmt = '%Y%m%d%H%M%S'
    now_str = datetime.datetime.now().strftime(fmt)

    dirstr = './GeneratedMusic/Music'+now_str+'.midi'
    song = stream.Stream()


    for gen_note in note_list:
        a_note = note.Note(str(gen_note[0])+str(gen_note[1]))
        a_note.duration.quarterLength = float(gen_note[2])
        song.append(a_note)
    song.write('midi', fp=dirstr)

def get_bach_midi():
    upperBound,lowerBound = 0,1000
    p_dict = {}
    d_dict = {}
    data = []
    try:
        for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='filename'):
            s = corpus.parse(score)
            if len(s.parts)==4:
                s = get_chords(s)
                song = []
                for c in s:
                    temp_chord = []
                    for n in c:
                        if p_dict.__contains__(str(n.pitch.midi))==False:
                            p_dict.__setitem__(str(n.pitch.midi),len(p_dict))

                        if d_dict.__contains__(str(n.duration.quarterLength))==False:
                            d_dict.__setitem__(str(n.duration.quarterLength),len(d_dict))

                        temp_chord.append(p_dict[str(n.pitch.midi)])
                    temp_chord.append(d_dict[str(c[0].duration.quarterLength)])
                    song.append(temp_chord)
                data.append(song)
                '''
                for p in s.parts:
                    for n in p.flat.notes:
                        if n.pitch.midi<lowerBound:
                            lowerBound = n.pitch.midi
                        if n.pitch.midi>upperBound:
                            upperBound=n.pitch.midi
                '''
    except Exception as e:
        print(e)

    saveobj([lowerBound,upperBound],'./Files/PitchRange')
    saveobj(p_dict,'./Files/BachMidiPitchDict')
    saveobj(d_dict,'./Files/BachMidiDurationDict')
    saveobj(data,'./Files/BachMidiChords')

def save_midi(seq,path):
    mf = MIDIFile(4)   # only 1 track
    track = 1          # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, 120)

    # add some notes
    channel = 1
    volume = 100

    pitch = 60           # C4 (middle C)
    time = 0             # start on beat 0
    duration = 1         # 1 beat long
    mf.addNote(0, channel, pitch, time, duration, volume)

    pitch = 20          # E4
    time = 2             # start on beat 2
    duration = 1         # 1 beat long
    mf.addNote(2, channel, pitch, time, duration, volume)

    pitch = 67           # G4
    time = 4             # start on beat 4
    duration = 1         # 1 beat long
    mf.addNote(3, channel, pitch, time, duration, volume)

    # write it to disk
    with open("output.mid", 'wb') as outf:
        mf.writeFile(outf)

if __name__ == '__main__':
    print("utildata.py main")
    #m = music(300)
    #a = loadobj('./Files/Octave')
    #print(a)
    #print(len(a))
    # create your MIDI object
