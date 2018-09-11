'''
LSTM-RBM model to generate music
'''

import tensorflow as tf
from tqdm import tqdm
import utildata as ud
import numpy as np
from music21 import*
import gc


#Get music object(a set of songs represented in numeric format)
num_songs = 200
music_obj = ud.music(num_songs)


#LSTM-RBM HYPERPARAMETERS
num_timesteps = 4
n_visible = 4*len(music_obj.pitch_dict)*num_timesteps
n_hidden = int(0.75*n_visible)

#LSTM Hidden Layer size
n_hidden_lstm = 128

#Dynamic learning rate, adjusted during training process
lr = tf.placeholder(tf.float32)

#python values NOT tensors variable_
batch_size_ = 50
lr_ = 0.001


'''Model's variables
x:input vector      u:LSTM hidden unit      h:RBM hidden layer
v:RBM visible layer                         W:RBM shared weigth matrix
b:bias              u0:Initial LSTM state
'''

#Input (music to be fed into visible layer/LSTM hidden unit)
x = tf.placeholder(tf.float32,[None,n_visible],name="x")

#Weights for RBMs
W = tf.Variable(tf.random_normal([n_visible,n_hidden],0.01),name="W")

'''We communicate to the RBM the current state of the song here:
Wuh,bh,Wuv,bv are used in calculating the biases of the RBM '''
#Weights from LSTM hidden unit at timestep t-1 to RBM hidden layer at timestep t
Wuh = tf.Variable(tf.random_normal([n_hidden_lstm,n_hidden],0.00001),name="Wuh")

#Bias from LSTM hidden unit at t-1 to RBM hidden layer at t
bh = tf.Variable(tf.zeros([1,n_hidden],tf.float32),name="bh")

#Weights from LSTM hidden unit at timestep t-1 to RBM visible layer at timestep t
Wuv = tf.Variable(tf.random_normal([n_hidden_lstm,n_visible],0.0001),name="Wuv")

#Bias from LSTM hidden unit at timestep t-1 to RBM visible later at timestep t
bv = tf.Variable(tf.zeros([1,n_visible],tf.float32),name="bv")

'''For the LSTM hidden unit in determining it's cell state and external state'''
#Weigths from RBM visible layer at timestep t to LSTM hidden unit at timestep t
Wvu = tf.Variable(tf.random_normal([4,n_visible,n_hidden_lstm],0.0001),name="Wvu")

#Weights from LSTM hidden unit at timestep t-1 to LSTM hidden unit at timestep at t
Wuu = tf.Variable(tf.random_normal([4,n_hidden_lstm,n_hidden_lstm],0.0001),name="Wuu")

#Bias from LSTM hidden layer at t-1 to LSTM hidden layer a t
bu = tf.Variable(tf.zeros([4,n_hidden_lstm],tf.float32),name="bu")

#Initial LSTM state
u0 = tf.Variable(tf.zeros([1,n_hidden_lstm],tf.float32),name="u0")
c0 = tf.Variable(tf.zeros([1,n_hidden_lstm],tf.float32),name="c0")

'''BIASES FOR RBM'''
#bias from visible -> hidden for RBM at timestep t
bv_t = tf.Variable(tf.ones([batch_size_,n_visible],tf.float32),name="bv_t")

#bias from hidden-> visible for RBM at timestep t
bh_t = tf.Variable(tf.ones([batch_size_,n_hidden],tf.float32),name="bh_t")

#TENSOR: batch_size
batch_size = tf.shape(x)[0]

#Reshaping bias matrices to be same size as the batch_size
tf.assign(bh_t,tf.tile(bh_t,[batch_size,1]))
tf.assign(bv_t,tf.tile(bv_t,[batch_size,1]))


'''Functions used within tf.Scan() to unroll the reccurrence'''
def lstm_recurrence(prev_t,xt):
    xt = tf.reshape(xt,[1,n_visible])

    #Two states in LSTM internal cell state(ct) and external state/output(st)
    #get previous states
    st_1,ct_1 = prev_t[0],prev_t[1]

    #Input layer:decides if new information is relevant then lets it in
    i = tf.sigmoid(tf.matmul(xt,Wvu[0])+tf.matmul(st_1,Wuu[0])+bu[0])
    #forget layer:gets rid of irrelevant information
    f = tf.sigmoid(tf.matmul(xt,Wvu[1])+tf.matmul(st_1,Wuu[1])+bu[1])
    #output layer
    o = tf.sigmoid(tf.matmul(xt,Wvu[2])+tf.matmul(st_1,Wuu[2])+bu[2])
    #some layer
    g = tf.tanh(tf.matmul(xt,Wvu[3])+tf.matmul(st_1,Wuu[3])+bu[3])

    #update internal cell state
    ct = (ct_1*f)+(g*i)
    #update external state
    st = tf.tanh(ct)*o

    return [st,ct]

hidden_bias_recurrence = lambda _,st_1: tf.add(bh,tf.matmul(st_1,Wuh))
visible_bias_recurrence = lambda _,st_1: tf.add(bv,tf.matmul(st_1,Wuv))

lstm_state = tf.scan(lstm_recurrence,x,initializer=[u0,c0])
s_t,c_t = lstm_state[0],lstm_state[1]

bv_t = tf.reshape(tf.scan(visible_bias_recurrence,s_t,tf.zeros([1,n_visible],tf.float32)),[batch_size,n_visible])
bh_t = tf.reshape(tf.scan(hidden_bias_recurrence,s_t,tf.zeros([1,n_hidden],tf.float32)),[batch_size,n_hidden])

epochs = 100

#Convert music_obj into input sequences for training(reshape into timesteps)
def get_input_sequences(music_obj,num_timesteps,n_visible):
    visible_layer_inputs = []

    for song in music_obj.data: #Traverse each song in set
        for chord_index in range(0,len(song)-num_timesteps+1,1): #get timesteps of chords in song
            chord_set = song[chord_index:chord_index+num_timesteps]
            chord_encoding = np.zeros([n_visible]) #initialize 0xn_visible
            for chord_set_index in range(len(chord_set)):#Traverse chords in chord set
                for note_index in range(4): #4 notes in each chord
                    #Will flip bit in encoding from 0 to 1 if note pitch is played
                    offset = (note_index*len(music_obj.pitch_dict)) + (4*len(music_obj.pitch_dict)*chord_set_index)
                    chord_encoding[chord_set[chord_set_index][note_index][0]+offset] = 1.0
                    #temp_in.append(chord_set[chord_set_index][note_index])
                visible_layer_inputs.append(chord_encoding)

    #Clear unreferenced memory
    gc.collect()

    return visible_layer_inputs

'''
NEEDS FIXING
'''
def initialize_model(songs,sess,weights_path=None):
    '''Initialize the model by training a single RBM.
    The parameters learned are then saved and uses to initialize all RBMS in the LSTM-RBM model
    :param sess The current TensorFlow session
    '''

    saver = tf.train.Saver([W,Wuh,Wuv,Wvu,Wuu,bh,bv,bu,u0,c0])

    #Path given so weights are already saved then we just load them
    if weights_path:
        saver.restore(sess,weights_path)
    else:
        #Initialize variables
        sess.run(tf.global_variables_initializer())
        epochs = 50

        print("---Training Weights on a single RBM---")
        for epoch in tqdm(range(epochs)):
            for song in songs:
                sess.run(contrastive_divergence(k=1),feed_dict={x:song})
        saver.save(sess,weights_path)
    return sess


#Helper function for sampling from RBM probability distribution using Gibbs Sampling
sample = lambda prob_dist: tf.floor(prob_dist+tf.random_uniform(tf.shape(prob_dist),0,1))

#Gibbs sampling recursively from visible layer to hidden layer then back to visible layer
def gibbs_sample(x,W,bv,bh,k):
    def gibbs_step(i,k,xk):
        '''Perform a SINGLE gibbs step
        :param i: current loop iteration
        :param k: number of gibbs step to perform
        :param xk: The output sampled from RBM
        '''
        #Feed the input x into the visible layer
        v = xk
        #Forward propagation to sample hk from the hidden layer
        hk = sample(tf.sigmoid(tf.matmul(v,W)+bh))
        #Backpropgate to sample xk from the visible layer
        xk = sample(tf.sigmoid(tf.matmul(hk,tf.transpose(W))+bv))
        return i+1,k,xk

    #Run k-gibbs steps and return the sample
    [_,_,x_sample] = tf.while_loop(lambda i,n,*args: i < n, gibbs_step, [0,k,x],
                                parallel_iterations=1,back_prop=False)

    return tf.stop_gradient(x_sample)

def free_energy_cost(x,W,bv,bh,k):
    '''Calculate the loss of the model, since the RBM is an energy based model,
    We Calculate the free energy cost between input and sample
    '''

    x_sample = gibbs_sample(x,W,bv,bh,k)

    #Function to that returns free energy of v (visible layer)
    free_energy = lambda v: - tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(v,W)+bh)),1)-tf.matmul(v,tf.transpose(bv))

    #Loss is difference in free energy between the sample and the original
    cost = tf.reduce_mean(tf.subtract(free_energy(x),free_energy(x_sample)))

    return cost

def contrastive_divergence(k,lr=0.001):
    '''Run k steps of the contrastive divergence '''

    #Sample visible layer x
    x_sample = gibbs_sample(x,W,bv,bh,k)

    h = sample(tf.sigmoid(tf.matmul(x,W)+bh))

    h_sample = sample(tf.sigmoid(tf.matmul(x_sample,W)+bh))


    '''Update the weights and biases by using the difference
    '''
    lr = tf.constant(lr,tf.float32)
    batch_size = tf.cast(tf.shape(x)[0],tf.float32)
    dW = tf.multiply(lr/batch_size,tf.subtract(tf.matmul(tf.transpose(x),h),tf.matmul(tf.transpose(x_sample),h_sample)))
    dbv = tf.multiply(lr/batch_size,tf.reduce_sum(tf.subtract(x,x_sample),0,True))
    dbn = tf.multiply(lr/batch_size,tf.reduce_sum(tf.subtract(h,h_sample),0,True))

    return [W.assign_add(dW),bv.assign_add(dbv),bh.assign_add(dbh)]

def train(data,weights_filepath=None):
    '''Train LSTM-RBM model'''

    lstm_state = tf.scan(lstm_recurrence,x,initializer=[u0,c0])
    s_t, c_t = lstm_state[0],lstm_state[1]

    bh_t = tf.reshape(tf.scan(hidden_bias_recurrence,s_t,tf.zeros([1,n_hidden],tf.float32)),[batch_size,n_hidden])
    bv_t = tf.reshape(tf.scan(visible_bias_recurrence,s_t,tf.zeros([1,n_visible],tf.float32)),[batch_size,n_visible])

    saver = tf.train.Saver([W,Wuh,Wuv,Wvu,Wuu,bh,bv,bu,u0,c0])

    if weights_filepath:
        with tf.Session() as sess:
            saver.restore(sess,weights_filepath)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    gradients = optimizer.compute_gradients(free_energy_cost(x,W,bv_t,bh_t,15),[W,Wuh,Wuv,Wvu,Wuu,bh,bv,bu,u0,c0])

    # Clip gradients to avoid exploding gradients
    #gradients = [(tf.clip_by_value(grad, -5.0, 5.0), hyperpar) for grad, hyperpar in gradients]

    logs_dir = "./graphs"

    '''START TRAINING'''
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logs_dir,sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        #sess = initialize_model(data,sess)

        print("---STARTED TRAINING---")
        for epoch in tqdm(range(epochs)):
            loss_epoch = 0 #Track loss after each epoch
            for i in range(0,len(data),batch_size_):
                _,cost = sess.run([optimizer.apply_gradients(gradients),free_energy_cost(x,W,bv_t,bh_t,15)],
                                feed_dict={x:data[i:i+batch_size_],lr:lr_ if epoch <= 10 else lr_/(epoch-10)})
                loss_epoch += abs(cost)
            print("\nLoss",loss_epoch/len(data),"at epoch",epoch)


            saver.save(sess,"./TrainingData/LSTM-RBM/epoch"+str(epoch)+str(loss_epoch)+".ckpt")

    writer.close()

def compose(song_timesteps,prime_timesteps=80):

    def compose_(i,k,prev_t,primer,x,pred):

        st_1, ct_1 = prev_t[0],prev_t[1]

        bv_t = tf.add(bv,tf.matmul(st_1,Wuv))
        bh_t = tf.add(bh,tf.matmul(st_1,Wuh))

        x_out =gibbs_sample(primer,W,bv_t,bh_t,k=25)

        #Propagate through the LSTM using the current output 'x_out' and the LSTM hidden unit at t-1, st_1, ct_1

        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(x_out,Wvu[0])+tf.matmul(st_1,Wuu[0])+bu[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(x_out,Wvu[1])+tf.matmul(st_1,Wuu[1])+bu[1])
        #output layer
        o = tf.sigmoid(tf.matmul(x_out,Wvu[2])+tf.matmul(st_1,Wuu[2])+bu[2])
        #some layer
        g = tf.tanh(tf.matmul(x_out,Wvu[3])+tf.matmul(st_1,Wuu[3])+bu[3])

        #update internal cell state
        ct = (ct_1*f)+(g*i)
        #update external state
        st = tf.tanh(ct)*o

        #Append x_out to prediction
        pred = tf.concat(values=[pred,x_out],axis=0)

        return i+1,k,[st,ct],x_out,x,pred

    lstm_state = tf.scan(lstm_recurrence,x,initializer=[u0,c0])

    s_t,c_t = lstm_state[0],lstm_state[1]
    #s_t = s_t[int(np.floor(prime_timesteps/num_timesteps)),:,:]

    #lstm_state = [s_t,c_t]
    pred = tf.zeros([1,n_visible],tf.float32)

    ts = tf.TensorShape

    '''
    Repeat compose_ whilst i<n is True
    '''
    ts = tf.TensorShape  # To quickly define a TensorShape
    compose_loop_out = tf.while_loop(lambda i, n, *args: i < n, compose_, [tf.constant(1), tf.constant(song_timesteps), lstm_state,
                                     tf.zeros([1, n_visible], tf.float32), x, tf.zeros([1, n_visible], tf.float32)],
                                     shape_invariants=[ts([]), ts([]), [s_t.get_shape(),c_t.get_shape()], ts([1, n_visible]), x.get_shape(), ts([1,n_visible])])
    pred = compose_loop_out[10]
    return pred

def generate_music(weights_filepath):

    saver = tf.train.Saver([W,Wuh,Wuv,Wvu,Wuu,bh,bv,bu,u0])

    primer = np.zeros([1,n_visible])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,weights_filepath)

        for i in tqdm(range(1)):
            generated_music = sess.run(generate(30),feed_dict={x:primer})

            note_list = []
            for k in generated_music:
                for n in range(0,len(k)-len(music_obj.pitch_dict)+1,len(music_obj.pitch_dict)):
                    note_list.append(np.argmax(k[n:n+len(music_obj.pitch_dict)]))


            inv_pitch = {pitch_num: pitch_name for pitch_name,pitch_num in music_obj.pitch_dict.items()}
            import datetime
            fmt = '%Y%m%d%H%M%S'
            now_str = datetime.datetime.now().strftime(fmt)

            dirstr ="./GeneratedMusic/LSTM_RBM_SONG"+now_str+".midi"
            song = stream.Stream()

            for i in range(0,len(note_list)-4+1,4):
                chord_list = note_list[i:i+4]
                for j in range(4):
                    chord_list[j] = inv_pitch[chord_list[j]]
                song.append(chord.Chord(chord_list))
            song.write('midi',fp=dirstr)


def generate_recurrence(count, k, prev_t, primer, x, music):
    #This function builds and runs the gibbs steps for each RBM in the chain to generate music
    #Get the bias vectors from the current state of the RNN

    st_1, ct_1 = prev_t[0],prev_t[1]

    bv_t = tf.add(bv, tf.matmul(st_1, Wuv))
    bh_t = tf.add(bh, tf.matmul(st_1, Wuh))

    #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
    x_out = gibbs_sample(primer, W, bv_t, bh_t, k=25)

    #Update the RNN hidden state based on the musical output and current hidden state.

    #Input layer:decides if new information is relevant then lets it in
    i = tf.sigmoid(tf.matmul(x_out,Wvu[0])+tf.matmul(st_1,Wuu[0])+bu[0])
    #forget layer:gets rid of irrelevant information
    f = tf.sigmoid(tf.matmul(x_out,Wvu[1])+tf.matmul(st_1,Wuu[1])+bu[1])
    #output layer
    o = tf.sigmoid(tf.matmul(x_out,Wvu[2])+tf.matmul(st_1,Wuu[2])+bu[2])
    #some layer
    g = tf.tanh(tf.matmul(x_out,Wvu[3])+tf.matmul(st_1,Wuu[3])+bu[3])

    #update internal cell state
    ct = (ct_1*f)+(g*i)
    #update external state
    st = tf.tanh(ct)*o
    #Add the new output to the musical piece
    music = tf.concat([music, x_out],0)

    return count+1, k, [st_1,ct], x_out, x, music


def generate(num,prime_length=num_timesteps-1):
    """This function handles generating music. This function is one of the outputs of the build_rnnrbm function
        Args:
        num (int): The number of timesteps to generate
        x (tf.placeholder): The data vector. We can use feed_dict to set this to the music primer.
        size_bt (tf.float32): The batch size
        u0 (tf.Variable): The initial state of the RNN
        n_visible (int): The size of the data vectors
        prime_length (int): The number of timesteps into the primer song that we use befoe beginning to generate music
        Returns:
        The generated music, as a tf.Tensor
        """
    lstm_state = tf.scan(lstm_recurrence, x, initializer=[u0,c0])
    Uarr = lstm_state[0]
    U = Uarr[int(np.floor(prime_length/num_timesteps)), :, :]
    ts = tf.TensorShape
    [_, _, _, _, _, music] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                            generate_recurrence, [tf.constant(1), tf.constant(num), [U,lstm_state[1]],
                                            tf.zeros([1, n_visible], tf.float32), x,tf.zeros([1, n_visible],  tf.float32)],
                                            shape_invariants=[ts([]), ts([]), [U.get_shape(),lstm_state[1].get_shape()], ts([1, n_visible]), x.get_shape(), ts([None, n_visible])])
    return music

#generate_music("TrainingData\LSTM-RBM\epoch98536.010925292969.ckpt")
input_sequences = get_input_sequences(music_obj,num_timesteps,n_visible)
train(np.array(input_sequences))
