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

import matplotlib.pyplot as plt

dataset = ud.get_melody(ud.loadobj('./Files/BachDataEncoded'))
num_timesteps = 16

#Create input sequences and corresponding output sequences(one_hot categorical)
def get_sequences(dataset):
    input_sequences = []
    output_sequences = []
    #Traverse each song
    for song in dataset:
        #Traverse each (encoded) note in song
        for note_index in range(0,len(song)-num_timesteps,1):
            #input at time t (timesteps of notes)
            input_sequences.append(song[note_index:note_index+num_timesteps])
            #output at time t (a single note)
            output_sequences.append(song[note_index+num_timesteps])

    return input_sequences,output_sequences

#One-hot encode output
def one_hot(output_sequences):
    return np_utils.to_categorical(output_sequences)

#Standerdize input sequences for neural net
def normalize(input_sequences,output_size):
    num = len(input_sequences)
    input_sequences = np.reshape(input_sequences,(num,num_timesteps,1))
    normalized_input = input_sequences/float(output_size)
    return normalized_input

def train_network():
    input_sequences,output_sequences = get_sequences(dataset)
    output_sequences = one_hot(output_sequences)
    input_sequences = normalize(input_sequences,output_sequences.shape[1])

    model = create_network(input_sequences,output_sequences.shape[1])
    train(model,input_sequences,output_sequences)

def create_network(network_input,output_size):
    model = Sequential()

    model.add(LSTM(
    512,
    input_shape=(network_input.shape[1],network_input.shape[2])
    ))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])

    return model

def train(model,network_input,network_output):
    filepath = "./TrainingData/KERAS-LSTM/SHALLOWweights-{epoch:02d}-{loss:.4f}-{metrics:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
    )
   # callbacks_list = [checkpoint]

    history = model.fit(network_input,network_output,epochs=250,batch_size=50)
    #print(history.history.keys())

    # Plot training
    plt.plot(history.history['categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    #plt.savefig('keras_lstm_16_4_layerACC.png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    #plt.savefig('keras_lstm_16_4_layerLOSS.png')

if __name__=='__main__':
    train_network()
