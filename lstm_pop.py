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

dataset = ud.get_melody(ud.loadobj('pop_dataset'))
encoding_dict = ud.loadobj('encoding')

num_timesteps = 16

#Input and output sequences of dataset
def get_sequences(dataset):
    input_seq = []
    output_seq = []
    #Traverse each song in dataset
    for song in dataset:
        #Traverse each note (encoded) to form num_timesteps of note
        for note_index in range(0,len(song)-num_timesteps,1):
            input_seq.append(song[note_index:note_index+num_timesteps])
            output_seq.append(song[note_index+num_timesteps])

    input_seq,output_seq = transform_sequences(input_seq,output_seq)
    return input_seq,output_seq

#Normalize Input and transform output sequences into categorical form(one-hot encoding)
def transform_sequences(input_seq,output_seq):
    num = len(input_seq)
    input_seq = np.reshape(input_seq,(num,num_timesteps,1))
    input_seq = input_seq/float(len(encoding_dict))
    output_seq = np_utils.to_categorical(output_seq)
    return input_seq,output_seq

def train_network():
    input_seq,output_seq = get_sequences(dataset)
    #model = create_network(input_seq,output_seq.shape[1])
    #train(model,input_seq,output_seq)
    print(output_seq.shape[1])

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
    model.add(LSTM(512))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    return model

def train(model,network_input,network_output):
    filepath = "./TrainingData/KERAS-LSTM-POP/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input,network_output,epochs=2000,batch_size=25,callbacks=callbacks_list)

if __name__=='__main__':
    #train_network()
