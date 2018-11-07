import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import utildata as ud

num_timesteps = 16
encoding_dict = ud.loadobj('encoding')
pitch_dict = ud.loadobj('pitch')
duration_dict = ud.loadobj('duration')

output_size = len(encoding_dict)-2 #FIX THIS

def get_primer(num_sequences):
    primer = []
    for i in range(num_sequences):
        temp = []
        for j in range(num_timesteps): #num_timesteps per sequence
            temp.append(np.random.randint(0,len(encoding_dict)-3))
        primer.append(temp)
    return primer

def generate():
    network_input = get_primer(10)
    model = create_network(148)
    prediction_output = generate_notes(model,network_input)
    print(prediction_output)
    create_midi(prediction_output)

def create_network(output_size):
    model = Sequential()

    model.add(LSTM(
    512,
    input_shape=(num_timesteps,1),
    return_sequences=True
    ))
    model.add(Dropout(0.5))
    model.add(LSTM(1024,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(output_size)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

    #Load the weights
    model.load_weights('./TrainingData/KERAS-LSTM-POP/weights-251-0.1487.hdf5')

    return model

def generate_notes(model,network_input):
    #pick a random sequence from the input
    start = np.random.randint(0,len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    for i in range(num_timesteps*10):
        prediction_input = np.reshape(pattern,(1,len(pattern),1))
        prediction_input = prediction_input/(float(output_size)+2) #FIX THIS

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

    inv_encoding = {val: key for key,val in encoding_dict.items()}
    inv_pitch = {p_num: p_name for p_name,p_num in pitch_dict.items()}
    inv_duration = {d_num: d_name for d_name,d_num in duration_dict.items()}

    import datetime
    fmt = '%Y%m%d%H%M%S'
    now_str = datetime.datetime.now().strftime(fmt)

    dirstr ="./GeneratedMusic/KERAS_LSTM_POP_MELODY"+now_str+".midi"
    song = stream.Stream()

    for element in prediction_output:
        #get encoding(pitch;duration)
        pd = inv_encoding[element].split(';')
        p = inv_pitch[int(pd[0])]
        d = inv_duration[int(pd[1])]

        a_note = note.Note(str(p))
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
