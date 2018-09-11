import sys
sys.path.append(r'C:\Users\Creative\Documents\GitHub\music21')
from music21 import*
import numpy as np
import json

class music(object):
    def __init__(self,n=1):
        self.data = getRawData(n)
        #self.data[0].show()
        #Used to represent musical features numerical
        self.pitch_dict ={}
        self.duration_dict = {}
        self.octave_dict ={}

        self.song_length = 1000
        for i in range(len(self.data)):
            self.data[i] = get_chords(self.data[i])#get_chords(standerdizeMusic(self.data[i]))
            if len(self.data[i]) < self.song_length:
                self.song_length = len(self.data[i])

        self.populate_dict()
        self.store_dict()

        self.batch_number = 0

    '''
    Populate dictionaries to switch to and from numerical represenation of a musical note
    to a music21 Note obj.
    '''
    def populate_dict(self):
        #Assign a number to each unique feature
        for song_index in range(len(self.data)):
            for chord_index in range(len(self.data[song_index])):
                for note_index in range(4): #single note of a chord, 4 notes (parts)
                    if self.pitch_dict.__contains__(self.data[song_index][chord_index][note_index].name)==False:
                        self.pitch_dict.__setitem__(self.data[song_index][chord_index][note_index].name,len(self.pitch_dict))

                    if self.duration_dict.__contains__(self.data[song_index][chord_index][note_index].duration.quarterLength)==False:
                        self.duration_dict.__setitem__(self.data[song_index][chord_index][note_index].duration.quarterLength,len(self.duration_dict))

                    if self.octave_dict.__contains__(self.data[song_index][chord_index][note_index].octave)==False:
                        self.octave_dict.__setitem__(self.data[song_index][chord_index][note_index].octave,len(self.octave_dict))

                    #Convert Note obj to numerical format
                    self.data[song_index][chord_index][note_index] = [
                    self.pitch_dict[self.data[song_index][chord_index][note_index].name],
                    self.octave_dict[self.data[song_index][chord_index][note_index].octave],
                    self.duration_dict[self.data[song_index][chord_index][note_index].duration.quarterLength]
                    ]

    '''
    Save represenation of a note during training for testing
    '''
    def store_dict(self):
        with open('./Files/pitches.json', 'w') as fp:
            json.dump(self.pitch_dict, fp)
        with open('./Files/durations.json', 'w') as fp:
            json.dump(self.duration_dict, fp)
        with open('./Files/octaves.json', 'w') as fp:
            json.dump(self.octave_dict, fp)

    '''
    Returns [batch_size] X batch (input, labels not needed for RBM/VAE)
    [batch_size] of training examples, that is [batch_size] musical pieces
    '''
    def next_batch(self,batch_size):

        #Ensure batch size isn't larger than size of dataset
        if batch_size>len(self.data):
            batch_size = len(self.data)

        #Batch number cannot be past last index, reset
        if self.batch_number>=len(self.data):
            self.batch_number = 0

        #Batch size and batch number is valid but we might not have enough batches left
        if self.batch_number+batch_size>len(self.data):
            batch_size = (self.batch_number+batch_size)-len(self.data)

        batch_x = []
        for i in range(batch_size):
            batch_x.append(self.data[i+self.batch_number])

        self.batch_number += batch_size

        return batch_x


'''
Return stream objects of music
 - 4-part chorales
 - 4 beats per bar
'''
def getRawData(amount):
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

'''
Transpose music to A-minor or C-major
Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
'''
def standerdizeMusic(score):
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

'''
Returns an array with each chord
'''
def get_chords(song):
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


'''
Concatenate notes into a music21 stream obj and save piece as midi file
'''
def create_music(note_list):
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

if __name__ == '__main__':
    print("utildata.py main")
    #s = getRawData(5)[4]
    #print(s.Chords)
    #cMinor = chord.Chord(["C4","G4","E-5"])
    #cMinor.show()
    import tensorflow as tf
    a = [1,1,1,1,1,1,1,1,1,1]
    print(a)
    b = tf.to_float(a)
    print(a)
