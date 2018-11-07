import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.python.ops import control_flow_ops

#music21 library local to disk import
import sys
sys.path.append(r'C:\Users\Creative\Documents\GitHub\music21')
from music21 import*

import utildata as ud

num_songs = 200
dataobj = ud.music(num_songs)

num_timesteps = 15
n_visible = 4*len(dataobj.pitch_dict)*num_timesteps#*len(music_obj.octave_dict)*len(music_obj.duration_dict)
n_hidden = int(0.75*n_visible)

visible_layer_inputs = []
for song in dataobj.data:
    for chord_index in range(0,len(song)-num_timesteps+1,num_timesteps):
        chord_set = song[chord_index:chord_index+num_timesteps]
        chord_encoding = np.zeros([n_visible])
        for chord_index in range(len(chord_set)):
            for note_index in range(4):
                offset = (note_index*len(dataobj.pitch_dict)) + (4*len(dataobj.pitch_dict)*chord_index)
                chord_encoding[chord_set[chord_index][note_index][0]+offset] = 1.0

        visible_layer_inputs.append(chord_encoding)

batch_size = 50

#Shape data
music_set = np.array(visible_layer_inputs)

num_epochs = 100
lr = tf.constant(0.001,tf.float32)

x  = tf.placeholder(tf.float32, [None, n_visible], name="x") #The placeholder variable that holds our data
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") #The weight matrix that stores the edge weights
bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh")) #The bias vector for the hidden layer
bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv")) #The bias vector for the visible layer


#This function lets us easily sample from a vector of probabilities
def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


#This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

        '''
                    cond = lambda i, h_mean, h_sample, v_mean, v_sample: i < step_every
                    body = lambda i, h_mean, h_sample, v_mean, v_sample: (i+1, ) + rbm.gibbs_vhv(v_sample)
                    i, h_mean, h_sample, v_mean, v_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros([n_chains, n_hidden]),
                                                                        tf.zeros([n_chains, n_hidden]), tf.zeros(tf.shape(persistent_v_chain)), persistent_v_chain])
        '''
    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                             gibbs_step, [ct, tf.constant(k), x])
    #This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

### Training Update Code
# Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
#The sample of x
x_sample = gibbs_sample(1)
#The sample of the hidden nodes, starting from the visible state of x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
#The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

#Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
#When we do sess.run(updt), TensorFlow will run all 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


### Run the graph!
# Now it's time to start a session and run the graph!
def train():
    with tf.Session() as sess:
        #First, we train the model
        #initialize the variables of the model
        init = tf.global_variables_initializer()
        sess.run(init)
        #Run through all of the training data num_epochs times
        for epoch in tqdm(range(num_epochs)):
            #for song in songs:
                #The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
                #Here we reshape the songs so that each training example is a vector with num_timesteps x 2*note_range elements
            #    song = np.array(song)
            #    song = song[:np.floor(song.shape[0]/num_timesteps)*num_timesteps]
            #    song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
                #Train the RBM on batch_size examples at a time
            #    for i in range(1, len(song), batch_size):
            for i in range(0,len(music_set),batch_size):
                tr_x = music_set[i:i+batch_size]
                sess.run(updt, feed_dict={x: tr_x})

        saver = tf.train.Saver()
        saver.save(sess,"./TrainingData/Basic-RBM/w.ckpt")
        sess.close()



def generate_music():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess,"./TrainingData/Basic-RBM/w.ckpt")

        #Now the model is fully trained, so let's make some music!
        #Run a gibbs chain where the visible nodes are initialized to 0
        sample = gibbs_sample(1000).eval(session=sess, feed_dict={x: np.zeros((1, n_visible))})
        for i in range(sample.shape[0]):
            if not any(sample[i,:]):
                continue

        note_list =[]
        for n in range(0,len(sample[0])-len(dataobj.pitch_dict)+1,len(dataobj.pitch_dict)):
            note_list.append(np.argmax(sample[0][n:n+len(dataobj.pitch_dict)]))


        inv_pitch = {pitch_num: pitch_name for pitch_name,pitch_num in dataobj.pitch_dict.items()}
        import datetime
        fmt = '%Y%m%d%H%M%S'
        now_str = datetime.datetime.now().strftime(fmt)

        dirstr ='C:/Users/Creative/Desktop/Project/GeneratedMusic/RBM_BASIC_SONG'+now_str+'.midi'
        song = stream.Stream()

        for i in range(0,len(note_list)-4+1,4):
            chord_list = note_list[i:i+4]
            for j in range(4):
                chord_list[j] = inv_pitch[chord_list[j]]
            song.append(chord.Chord(chord_list))
        song.write('midi',fp=dirstr)

#train()
generate_music()
