import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import utildata as ud

class MusicGenerator(object):
    def __init__(self,music_obj,num_timesteps):
        self.music_obj = music_obj
        self.num_timesteps = num_timesteps

        #CHANGE THIS TO GET THIS FROM CSV FILE(TRAINING PARAMETERS)
        self.num_pitches = len(self.music_obj.pitch_dict)
        self.num_durations = len(self.music_obj.duration_dict)

        self.pitch_inputs,self.duration_inputs = self.get_sequences()

        self.pitch_model = self.create_pitch_network()
        self.duration_model = self.create_duration_network()

    def get_sequences(self):
        p_inputs = []
        d_inputs = []

        #Traverse each song
        for song_index in range(len(self.music_obj.data)):
            for chord_index in range(0,len(self.music_obj.data[song_index])-self.num_timesteps,1):
                chord_set = self.music_obj.data[song_index][chord_index:chord_index+self.num_timesteps]
                temp_p = []
                temp_d = []
                for chord in chord_set:
                    temp_p.append(chord[0][0])
                    temp_d.append(chord[0][2])
                p_inputs.append(temp_p)
                d_inputs.append(temp_d)
        return p_inputs,d_inputs

    def create_pitch_network(self):
        model = Sequential()
        model.add(LSTM(
        256,
        input_shape=(self.num_timesteps,1),
        return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512,return_sequences=True))
        model.add(Dropout(0,3))
        model.add(LSTM(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_pitches))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

        #LOAD THE WEIGHTS CONSTRUCTED FROM TRAINING
        model.load_weights('TrainingData\KERAS-LSTM\weights-200-0.6579.hdf5')

        return model

    def create_duration_network(self):
        model = Sequential()
        model.add(LSTM(
        256,
        input_shape=(self.num_timesteps,1),
        return_sequences=True,
        ))
        model.add(Dropout(0,3))
        model.add(LSTM(512,return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_durations))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

        #LOAD THE WEIGHTS
        model.load_weights('TrainingData\KERAS-LSTM-DURATION\weights-195-0.6267.hdf5')

        return model

    def predict_music(self,num_notes):
        #pick a random sequence from the input
        start = np.random.randint(0,len(self.pitch_inputs)-1)

        p_pattern = self.pitch_inputs[start]
        d_pattern = self.duration_inputs[start]
        print(d_pattern)
        prediction_output = []

        for i in range(num_notes):

            prediction_p_input = np.reshape(p_pattern,(1,len(p_pattern),1))
            prediction_p_input = prediction_p_input/float(self.num_pitches)

            prediction_d_input = np.reshape(d_pattern,(1,len(d_pattern),1))
            prediction_d_input = prediction_d_input/float(self.num_durations)

            p_prediction = self.pitch_model.predict(prediction_p_input,verbose=0)
            d_prediction = self.duration_model.predict(prediction_d_input,verbose=0)

            p_index = np.argmax(p_prediction)
            d_index = np.argmax(d_prediction)
            print(d_prediction)
            prediction_output.append([p_index,d_index])

            p_pattern.append(p_index)
            d_pattern.append(d_index)

            p_pattern = p_pattern[1:len(p_pattern)]
            d_pattern = d_pattern[1:len(d_pattern)]

        return prediction_output

if __name__ == "__main__":
    musicobj = ud.music(10)
    mg = MusicGenerator(musicobj,4)
    predicted_music = mg.predict_music(50)
    print(predicted_music)
    print(musicobj.duration_dict)
