import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import utildata as ud

num_timesteps = 16
output_size = 79 #FIX THIS HACK

def generate():
    model = create_network(output_size)
    prediction_output = generate_notes(model)
    print(prediction_output)
    create_midi(prediction_output)

def create_network(output_size):
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
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    #Load the weights
    model.load_weights('TrainingData\KERAS-LSTM\weights-199-0.9446.hdf5')

    return model

'''
Generate melody from scratch (sort of)
Initial input is num_timesteps of random notes
'''
def generate_notes(model):
    #Generate Initial (random) input
    pattern = []
    for i in range(num_timesteps):
        pattern.append(np.random.randint(0,output_size))

    prediction_output = []

    for i in range(500):
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
    pitch_dict = ud.loadobj('./Files/BachPitch')
    duration_dict = ud.loadobj('./Files/BachDuration')
    encoding_dict = ud.loadobj('./Files/BachEncoding')
    
    inv_encoding = {val: key for key,val in encoding_dict}
    inv_pitch = {pitch: pnum for pnum,pitch in pitch_dict}
    inv_duration = {duration: dnum for dnum,duration in duration_dict}

    import datetime
    fmt = '%Y%m%d%H%M%S'
    now_str = datetime.datetime.now().strftime(fmt)

    dirstr ="./GeneratedMusic/KERAS_LSTM_SONG"+now_str+".midi"
    song = stream.Stream()

    for element in prediction_output:
        pd = inv_encoding[element].split(';')
        p = inv_pitch[int(pd[0])]
        d = inv_duration[int(pd[1])]

        a_note = note.Note(p)
        try:
            a_note.duration.quarterLength = float(d)
        except:
            tmp = d.split('/')
            a_note.duration.quarterLength = float(float(tmp[0])/float(tmp[1]))
            pass
        song.append(a_note)
    song.write('midi',fp=dirstr)

if __name__ == '__main__':
    generate()
