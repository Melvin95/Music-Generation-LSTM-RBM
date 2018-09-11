import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import utildata as ud

num_songs = 15
musicobj = ud.music(num_songs)
num_timesteps = 4
output_size = len(musicobj.pitch_dict)

def generate():
    network_input = get_sequences()
    model = create_network(network_input)
    prediction_output = generate_notes(model,network_input)
    print(prediction_output)
    create_midi(prediction_output)

#Create input sequences
def get_sequences():
    input_sequences = []
    #output_sequences = []
    #Traverse each song
    for song_index in range(len(musicobj.data)):
        #Traverse each chord
        for chord_index in range(0,len(musicobj.data[song_index])-num_timesteps,1):
            chord_set = musicobj.data[song_index][chord_index:chord_index+num_timesteps]
            #output_sequences.append([musicobj.data[song_index][chord_index+num_timesteps][0][0]])
            temp_in = []
            for chord in chord_set:
                temp_in.append(chord[0][0])
            input_sequences.append(temp_in)

    #num = len(input_sequences)
    #input_sequences = np.reshape(input_sequences,(num,num_timesteps,1))
    return input_sequences #,np_utils.to_categorical(output_sequences)

def create_network(network_input):
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(num_timesteps,1),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    #Load the weights
    model.load_weights('TrainingData\KERAS-LSTM\weights-199-0.9446.hdf5')

    return model

def generate_notes(model,network_input):
    #pick a random sequence from the input
    start = np.random.randint(0,len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    for i in range(50):
        prediction_input = np.reshape(pattern,(1,len(pattern),1))
        prediction_input = prediction_input/float(output_size)

        prediction = model.predict(prediction_input,verbose=0)

        index = np.argmax(prediction)
        prediction_output.append(index)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

'''
Create a midi file from a list of integers
integers represent a certain pitch
'''
def create_midi(prediction_output):

    inv_pitch = {pitch_num: pitch_name for pitch_name,pitch_num  in musicobj.pitch_dict.items()}

    import datetime
    fmt = '%Y%m%d%H%M%S'
    now_str = datetime.datetime.now().strftime(fmt)

    dirstr ="./GeneratedMusic/KERAS_LSTM_SONG"+now_str+".midi"
    song = stream.Stream()

    for some_note in prediction_output:
        song.append(note.Note(some_note))
    song.write('midi',fp=dirstr)

if __name__ == '__main__':
    generate()
