from music21 import*
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import utildata as ud
import numpy as np

num_songs = 10
musicobj = ud.music(num_songs)
num_timesteps = 4

#Create input sequences and corresponding output sequences(one_hot categorical)
def get_sequences():
    input_sequences = []
    output_sequences = []
    #Traverse each song
    for song_index in range(len(musicobj.data)):
        #Traverse each chord
        for chord_index in range(0,len(musicobj.data[song_index])-num_timesteps,1):
            chord_set = musicobj.data[song_index][chord_index:chord_index+num_timesteps]
            output_sequences.append(musicobj.data[song_index][chord_index+num_timesteps][0][0])
            temp_in = []
            for chord in chord_set:
                temp_in.append(chord[0][0])
            input_sequences.append(temp_in)

    num = len(input_sequences)
    input_sequences = np.reshape(input_sequences,(num,num_timesteps,1))
    input_sequences = input_sequences/float(len(musicobj.pitch_dict))
    return input_sequences,np_utils.to_categorical(output_sequences)


def train_network():
    inputseq,outputseq = get_sequences()

    model = create_network(inputseq)

    train(model,inputseq,outputseq)

def create_network(network_input):
    model = Sequential()

    model.add(LSTM(
    256,
    input_shape=(network_input.shape[1],network_input.shape[2]),
    return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(len(musicobj.pitch_dict)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    return model

def train(model,network_input,network_output):
    filepath = "./TrainingData/KERAS-LSTM/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input,network_output,epochs=200,batch_size=75,callbacks=callbacks_list)

if __name__=='__main__':
    train_network()
