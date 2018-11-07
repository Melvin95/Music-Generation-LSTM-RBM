from music21 import*
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten

import utildata as ud
import numpy as np

dataset = ud.loadobj('./Files/BachChords')
num_timesteps = 16

#Create input sequences and corresponding output sequences(one_hot categorical)
def get_sequences(dataset):
    input_sequences = []
    output_sequences = []
    #Traverse each song
    for song in dataset:
        for i in range(0,len(song)-(num_timesteps+1),1):
            timestep_set = song[i:i+num_timesteps]
            temp_in = []
            for something in timestep_set:
                note = something[0]
                time = something[len(something)-1]
                temp_in.append([note,time])
            input_sequences.append(temp_in)
            note_out = song[i+num_timesteps][0]
            time_out = song[i+num_timesteps][len(song[i+num_timesteps])-1]
            output_sequences.append([note_out,time_out])
    return input_sequences,output_sequences

#One-hot encode output
def one_hot(output_sequences):
    return np_utils.to_categorical(output_sequences)

#Standerdize input sequences for neural net
def normalize(input_sequences,output_size):
    num = len(input_sequences)
    input_sequences = np.reshape(input_sequences,(num,num_timesteps,2))
    normalized_input = input_sequences/float(output_size)
    return normalized_input

def train_network():
    input_sequences,output_sequences = get_sequences(dataset)
    output_sequences = one_hot(output_sequences)
    print(output_sequences)
    input_sequences = normalize(input_sequences,output_sequences.shape[1])

    model = create_network(input_sequences,output_sequences.shape[1])
    train(model,input_sequences,output_sequences)

def create_network(network_input,output_size):
    model = Sequential()

    model.add(LSTM(
    512,
    input_shape=(network_input.shape[1],network_input.shape[2]),
    return_sequences=True
    ))
    model.add(Dropout(0.5))
    model.add(LSTM(1024,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(1024,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])

    return model

def train(model,network_input,network_output):
    filepath = "./TrainingData/KERAS-LSTM/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input,network_output,epochs=1000,batch_size=50,callbacks=callbacks_list)

if __name__=='__main__':
    train_network()
