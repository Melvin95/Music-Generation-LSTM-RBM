import utildata as ud
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from music21 import*

num_songs = 2
dataobj = ud.music(num_songs)
time_steps = 8

def get_sequences():
    input_sequences = []
    for song_index in range(len(dataobj.data)): #Traverse each song
        for chord_index in range(0,len(dataobj.data[song_index])-time_steps+1,time_steps): #Traverse chords in song
            chord_set = dataobj.data[song_index][chord_index:chord_index+time_steps]
            temp_in = []
            for chord in chord_set:
                temp_in.append([chord[0][0]])
            input_sequences.append(temp_in)
    return input_sequences

def categorize_output(seq):
    output_sequences = []
    output_sequences.append(np.zeros(len(dataobj.pitch_dict)))
    for i in range(len(seq)): #Traverse each sequence
        encoding = np.zeros(len(dataobj.pitch_dict))
        encoding[seq[i][0]] = 1.0
        output_sequences.append(encoding)
    return output_sequences

input_sequences = np.array(get_sequences())
output_sequences = categorize_output(input_sequences[1:])


X = tf.placeholder(tf.float32,shape=[None,time_steps,1])
Y = tf.placeholder(tf.float32,shape=[None,len(dataobj.pitch_dict)])

weights = {
    'out': tf.Variable(tf.random_normal([700,len(dataobj.pitch_dict)]))
}
biases = {
    'out':tf.Variable(tf.random_normal([len(dataobj.pitch_dict)]))
}

def RNN(x,weights,biases):

    x = tf.unstack(x,time_steps,1)

    lstm_cell = rnn.BasicLSTMCell(700,forget_bias=0.05)

    output,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)

    return tf.matmul(output[-1],weights['out'])+biases['out']

logits = RNN(X,weights,biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def train_network():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_writer = tf.summary.FileWriter( './logs/1/train', sess.graph)

        print("---STARTING TRAINING---")
        for epoch in range(5000):
            train_feed = {X:input_sequences,Y:output_sequences}

            t = sess.run(train_op,train_feed)
            print(epoch)

        saver = tf.train.Saver()
        saver.save(sess, "./TrainingData/LSTM/w.ckpt")
        incorrect = sess.run(loss_op,{X: input_sequences, Y: output_sequences})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * incorrect))
        sess.close()

#train_network()

def generate_network():
    generated_notes = []
    with tf.Session() as sess:
        print("GENERATING MUSIC")
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess,"./TrainingData/LSTM/w.ckpt")

        inv_pitch = {pitch_num: pitch_name for pitch_name,pitch_num in dataobj.pitch_dict.items()}
        #test = np.zeros([1,time_steps,1])
        test = np.reshape(np.array(input_sequences[1]),(1,time_steps,1))
        for c in range(50):
            predicted = sess.run(prediction,{X:test})
            pitch = inv_pitch[np.argmax(predicted)]
            generated_notes.append(pitch)
            test[0,c%time_steps] = np.argmax(predicted)
            print(test)
        sess.close()

        import datetime
        fmt = '%Y%m%d%H%M%S'
        now_str = datetime.datetime.now().strftime(fmt)

        dirstr ="./GeneratedMusic/LSTM_SONG"+now_str+".midi"
        song = stream.Stream()

        for i in range(len(generated_notes)):
            song.append(note.Note(generated_notes[i]))
        song.write('midi',fp=dirstr)

generate_network()
