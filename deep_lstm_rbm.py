'''
LSTM-RBM model to generate music
'''

import tensorflow as tf
from tqdm import tqdm
import utildata as ud
import numpy as np
from music21 import*
import gc
import random
import matplotlib.pyplot as plt
from midiutil.MidiFile import MIDIFile

class lstm_rbm(object):
    '''LSTM-RBM class'''
    def __init__(self,input=None,rbm_path=None,n_hidden=None,n_visible=None,num_timesteps=None,epochs=None,batch_size=None,pitch_dict=None,duration_dict=None,octave_dict=None,pitch_oct_dict=None):
        '''
        :param config: contains model's parameters(number of layers/timesteps/batch size/learning rate etc)
        :input: array of data
        '''
        #Music represented in an array
        self.dataset = input
        self.duration_dict = duration_dict
        self.pitch_dict = pitch_dict
        self.pitch_oct_dict = pitch_oct_dict
        self.octave_dict = octave_dict

        self.weights_path = rbm_path

        #LSTM-RBM HYPER-PARAMETERS
        self.num_timesteps = num_timesteps
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.epochs = epochs

        #LSTM hidden unit size (number of neurons/width)
        self.n_hidden_lstm = 512

        #Learning rate placeholder, adjusted during training
        self.lr = 0.001

        #Batch size and learning rate VALUES
        self.batch_size_ = batch_size
        self.lr_ = 0.001

        '''
        Define variables for model
        '''
        #Input tensor with shape [?,n_visible], for visible layer
        self.x = tf.placeholder(tf.float32,[None,self.n_visible],name="x")

        #RBM shared weights
        self.W = tf.Variable(tf.random_normal([self.n_visible,self.n_hidden],0.01),name="W")

        '''communicate sequence history to the RBM hidden layer(determine bh_t)'''
        #Weights from LSTM hidden unit a t-1 to RBM hidden layer at t
        self.Wuh = tf.Variable(tf.random_normal([self.n_hidden_lstm,self.n_hidden],0.000001),name="Wuh")
        #Bias from LSTM hidden unit a t-1 to RBM hidden layer at t
        self.bh = tf.Variable(tf.zeros([1,self.n_hidden],tf.float32),name="bh")

        '''communicate sequence history to the RBM visible layer(determine bv_t)'''
        #Weigths from LSTM hidden unit at t-1 to RBM visible layer a t
        self.Wuv = tf.Variable(tf.random_normal([self.n_hidden_lstm,self.n_visible],0.00001),name="Wuv")
        #Bias from LSTM hidden unit at t-1 to RBM visible layer a t
        self.bv = tf.Variable(tf.zeros([1,self.n_visible],tf.float32),name="bv")

        '''LSTM hidden unit variables'''
        #Weights of the input/music from RBM visible layer at t to LSTM hidden unit at t
        self.Wvu = tf.Variable(tf.random_normal([4,self.n_visible,self.n_hidden_lstm],0.0001),name="Wvu")
        #Weights between each LSTM hidden units through time
        self.Wuu = tf.Variable(tf.random_normal([4,self.n_hidden_lstm,self.n_hidden_lstm],0.0001),name="Wuu")
        #Bias for LSTM hidden units through time
        self.bu = tf.Variable(tf.zeros([4,self.n_hidden_lstm],tf.float32),name="bu")
        #Initialize LSTM with internal and external states
        self.u0 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="u0")
        self.c0 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="c0")

        '''RBM biases(bias for RBM at a particular time t)'''
        #Bias to propagate from visible->hidden for RBM t
        self.bv_t = tf.Variable(tf.ones([self.batch_size_,self.n_visible],tf.float32),name="bv_t")
        #Bias to propagate from hidden->visible for RBM t
        self.bh_t = tf.Variable(tf.ones([self.batch_size_,self.n_hidden],tf.float32),name="bh_t")

        #tensor of batch_size
        self.batch_size = tf.shape(self.x)[0]
        #Reshape bias matrices
        tf.assign(self.bh_t,tf.tile(self.bh_t,[self.batch_size,1]))
        tf.assign(self.bv_t,tf.tile(self.bv_t,[self.batch_size,1]))

        '''PARAMETERS FOR MULTI-LAYERED LSTM'''
        self.W1 = tf.Variable(tf.random_normal([4,self.n_hidden_lstm,self.n_hidden_lstm],0.0001),name="W1")
        self.U1 = tf.Variable(tf.random_normal([4,self.n_hidden_lstm,self.n_hidden_lstm],0.0001),name="U1")
        self.b1 = tf.Variable(tf.zeros([4,self.n_hidden_lstm],tf.float32),name="b1")
        self.u1 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="u1")
        self.c1 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="c1")

        self.W2 = tf.Variable(tf.random_normal([4,self.n_hidden_lstm,self.n_hidden_lstm],0.0001),name="W2")
        self.U2 = tf.Variable(tf.random_normal([4,self.n_hidden_lstm,self.n_hidden_lstm],0.0001),name="U2")
        self.b2 = tf.Variable(tf.zeros([4,self.n_hidden_lstm],tf.float32),name="b2")
        self.u2 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="u2")
        self.c2 = tf.Variable(tf.zeros([1,self.n_hidden_lstm],tf.float32),name="c2")

    def dropout(self,vec,prob=0.5):
        '''Simple Dropout Layer for LSTM'''
        mask = np.random.binomial(1,prob,size=self.n_hidden_lstm)/prob
        out = mask*vec
        return tf.reshape(out,[1,self.n_hidden_lstm])

    def deep_step(self,prev_t,xt):
        '''
        DEEP LSTM UNROLLED
        pretty poorly coded, hard-coded multiple layers so difficult to add layers if needed.
        Tried using a for-loop with tensor of shape [num_layers,4,state_size,state_size] but
        an exception kept being thrown about tf.float32 not supported.
        '''

        '''INPUT LAYER- RBM VISIBLE LAYER Wvu ---> LSTM UNIT Wuu'''
        xt = tf.reshape(xt,[1,self.n_visible])
        #Two states in LSTM internal cell state(ct) and external state/output(st)
        #get previous states
        st_1,ct_1= prev_t[0],prev_t[1]
        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(xt,self.Wvu[0])+tf.matmul(st_1,self.Wuu[0])+self.bu[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(xt,self.Wvu[1])+tf.matmul(st_1,self.Wuu[1])+self.bu[1])
        #output layer
        o = tf.sigmoid(tf.matmul(xt,self.Wvu[2])+tf.matmul(st_1,self.Wuu[2])+self.bu[2])
        #some layer
        g = tf.tanh(tf.matmul(xt,self.Wvu[3])+tf.matmul(st_1,self.Wuu[3])+self.bu[3])
        #update internal cell state
        ct = (ct_1*f)+(self.dropout(g)*i)
        #update external state
        st = tf.tanh(ct)*o

        '''HIDDEN LAYER 1'''
        sth_1,cth_1 = prev_t[2],prev_t[3]
        inp = st
        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(inp,self.W1[0])+tf.matmul(sth_1,self.U1[0])+self.b1[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(inp,self.W1[1])+tf.matmul(sth_1,self.U1[1])+self.b1[1])
        #output layer
        o = tf.sigmoid(tf.matmul(inp,self.W1[2])+tf.matmul(sth_1,self.U1[2])+self.b1[2])
        #some layer
        g = tf.tanh(tf.matmul(inp,self.W1[3])+tf.matmul(sth_1,self.U1[3])+self.b1[3])
        #update internal cell state
        cth = (cth_1*f)+(self.dropout(g)*i)
        #update external state
        sth = tf.tanh(cth)*o

        '''OUTPUT LAYER: LSTM --> RBM (bht/bvt)'''
        sth_1,cth_1 = prev_t[4],prev_t[5]
        inp = sth
        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(inp,self.W2[0])+tf.matmul(sth_1,self.U2[0])+self.b2[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(inp,self.W2[1])+tf.matmul(sth_1,self.U2[1])+self.b2[1])
        #output layer
        o = tf.sigmoid(tf.matmul(inp,self.W2[2])+tf.matmul(sth_1,self.U2[2])+self.b2[2])
        #some layer
        g = tf.tanh(tf.matmul(inp,self.W2[3])+tf.matmul(sth_1,self.U2[3])+self.b2[3])
        #update internal cell state
        cth2 = (cth_1*f)+(self.dropout(g)*i)
        #update external state
        sth2 = tf.tanh(cth)*o

        return [st,ct,sth,cth,sth2,cth2]

    def lstm_recurrence(self,prev_t,xt):
        '''
        SINGLE-LSTM UNROLLED
           Function to get values for LSTM hidden unit a t
           given (prev_t) LSTM unit at t-1 and (xt) current input
        '''
        xt = tf.reshape(xt,[1,self.n_visible])

        #Two states in LSTM internal cell state(ct) and external state/output(st)
        #get previous states
        st_1,ct_1= prev_t[0],prev_t[1]

        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(xt,self.Wvu[0])+tf.matmul(st_1,self.Wuu[0])+self.bu[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(xt,self.Wvu[1])+tf.matmul(st_1,self.Wuu[1])+self.bu[1])
        #output layer
        o = tf.sigmoid(tf.matmul(xt,self.Wvu[2])+tf.matmul(st_1,self.Wuu[2])+self.bu[2])
        #some layer
        g = tf.tanh(tf.matmul(xt,self.Wvu[3])+tf.matmul(st_1,self.Wuu[3])+self.bu[3])

        #update internal cell state
        ct = (ct_1*f)+(g*i)
        #update external state
        st = tf.tanh(ct)*o

        return [st,ct]

    def hidden_bias_recurrence(self,_,st_1):
        return tf.add(self.bh,tf.matmul(st_1,self.Wuh))

    def visible_bias_recurrence(self,_,st_1):
        return tf.add(self.bv,tf.matmul(st_1,self.Wuv))

    def sample(self,prob_dist):
        return tf.floor(prob_dist+tf.random_uniform(tf.shape(prob_dist),0,1))

    def initialize_model(self,sess):
        '''
        Pretrain RBM layer to initialize RBM parameters
        '''
        saver = tf.train.Saver([self.W,self.Wuh,self.Wuv,self.Wvu,self.Wuu,self.bh,self.bv,self.bu,self.u0,self.c0,self.u1,self.c1,self.u2,self.c2,self.W1,self.U1,self.b1,self.W2,self.U2,self.b2])

        #If model already initialized
        if self.weights_path:
            saver.restore(sess,self.weights_path)
        else:
            '''Contrastive Divergence Algorithm'''
            #Sample visible layer x
            x_sample = self.gibbs_sample(self.x,1)

            h = self.sample(tf.sigmoid(tf.matmul(self.x,self.W)+self.bh))

            h_sample = self.sample(tf.sigmoid(tf.matmul(x_sample,self.W)+self.bh))

            '''Update the weights and biases by using the difference
            '''
            batch_size = tf.cast(tf.shape(self.x)[0],tf.float32)
            dW = tf.multiply(self.lr_/batch_size,tf.subtract(tf.matmul(tf.transpose(self.x),h),tf.matmul(tf.transpose(x_sample),h_sample)))
            dbv = tf.multiply(self.lr_/batch_size,tf.reduce_sum(tf.subtract(self.x,x_sample),0,True))
            dbh = tf.multiply(self.lr_/batch_size,tf.reduce_sum(tf.subtract(h,h_sample),0,True))

            updt = [self.W.assign_add(dW),self.bv.assign_add(dbv),self.bh.assign_add(dbh)]

            #train on a single RBM
            sess.run(tf.global_variables_initializer())
            print("---Pretraining RBM Layer---")
            for epoch in tqdm(range(self.epochs)):
                for batch in tqdm(range(0,len(self.dataset)-self.batch_size_,self.batch_size_)):
                    batch_x = self.dataset[batch:batch+self.batch_size_]
                    sess.run(updt,feed_dict={self.x:batch_x})
            saver.save(sess,'./TrainingData/PREDEEPRBM/w.ckpt')
        return sess

    def gibbs_sample(self,x,k):
        def gibbs_step(i,k,xk):
            '''Perform a SINGLE gibbs step
            :param i: current loop iteration
            :param k: number of gibbs step to perform
            :param xk: The output sampled from RBM
            '''
            #Feed the input x into the visible layer
            v = xk
            #Forward propagation to sample hk from the hidden layer
            hk = self.sample(tf.sigmoid(tf.matmul(v,self.W)+self.bh))
            #Backpropgate to sample xk from the visible layer
            xk = self.sample(tf.sigmoid(tf.matmul(hk,tf.transpose(self.W))+self.bv))
            return i+1,k,xk

        #Run k-gibbs steps and return the sample
        [_,_,x_sample] = tf.while_loop(lambda i,n,*args: i < n, gibbs_step, [0,k,x],
                                    parallel_iterations=1,back_prop=False)

        return tf.stop_gradient(x_sample)

    def contrastive_divergence(self,k,lr=0.001):
        '''Run k steps of the contrastive divergence '''
        #Sample visible layer x
        x_sample = self.gibbs_sample(k)

        h = self.sample(tf.sigmoid(tf.matmul(self.x,self.W)+self.bh))

        h_sample = self.sample(tf.sigmoid(tf.matmul(x_sample,self.W)+self.bh))

        '''Update the weights and biases by using the difference
        '''
        batch_size = tf.cast(tf.shape(self.x)[0],tf.float32)
        dW = tf.multiply(self.lr_/batch_size,tf.subtract(tf.matmul(tf.transpose(self.x),h),tf.matmul(tf.transpose(x_sample),h_sample)))
        dbv = tf.multiply(self.lr_/batch_size,tf.reduce_sum(tf.subtract(self.x,x_sample),0,True))
        dbh = tf.multiply(self.lr_/batch_size,tf.reduce_sum(tf.subtract(h,h_sample),0,True))

        return [self.W.assign_add(dW),self.bv.assign_add(dbv),self.bh.assign_add(dbh)]

    def free_energy_cost(self,k):
        '''Calculate the loss of the model, since the RBM is an energy based model,
           Calculate the free energy cost between input and sample
        '''
        x_sample = self.gibbs_sample(k)

        #Function to that returns free energy of v (visible layer)
        free_energy = lambda v: - tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(v,self.W)+self.bh)),1)-tf.matmul(v,tf.transpose(self.bv))

        #Loss is difference in free energy between the sample and the original
        cost = tf.reduce_mean(tf.subtract(free_energy(self.x),free_energy(x_sample)))

        return cost

    def train(self):
        '''Unroll LSTM'''
        #lstm_state = tf.scan(self.lstm_recurrence,self.x,initializer=[self.u0,self.c0])
        lstm_states = tf.scan(self.deep_step,self.x,initializer=[self.u0,self.c0,self.u1,self.c1,self.u2,self.c2])
        s_t, c_t = lstm_states[4],lstm_states[5]

        '''Pass knowledge of LSTM to RBM, alter biases at time t'''
        self.bh_t = tf.reshape(tf.scan(self.hidden_bias_recurrence,s_t,tf.zeros([1,self.n_hidden],tf.float32)),[self.batch_size,self.n_hidden])
        self.bv_t = tf.reshape(tf.scan(self.visible_bias_recurrence,s_t,tf.zeros([1,self.n_visible],tf.float32)),[self.batch_size,self.n_visible])

        saver = tf.train.Saver([self.W,self.Wuh,self.Wuv,self.Wvu,self.Wuu,self.bh,self.bv,self.bu,self.u0,self.c0,self.u1,self.c1,self.u2,self.c2,self.W1,self.U1,self.b1,self.W2,self.U2,self.b2])

        '''Free-energy cost'''
        x_sample = self.gibbs_sample(self.x,15)
        #Function to that returns free energy of v (visible layer)
        free_energy = lambda v: - tf.reduce_sum(tf.log(1+tf.exp(tf.matmul(v,self.W)+self.bh)),1)-tf.matmul(v,tf.transpose(self.bv))
        #Loss is difference in free energy between the sample and the original
        freecost = tf.reduce_mean(tf.subtract(free_energy(self.x),free_energy(x_sample)))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients = optimizer.compute_gradients(freecost,[self.W,self.Wuh,self.Wuv,self.Wvu,self.Wuu,self.bh,self.bv,self.bu,self.u0,self.c0,self.u1,self.c1,self.u2,self.c2,self.W1,self.U1,self.b1,self.W2,self.U2,self.b2])
        appliedgrad = optimizer.apply_gradients(gradients)

        #tf_metric, tf_metric_update = tf.metrics.accuracy(self.x, x_sample,name="my_metric")
        #running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
        #running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        logs_dir = "./graphs"
        loss_list = []
        epoch_list = []
        '''START TRAINING'''
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logs_dir,sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)
            #Pretrain
            sess = self.initialize_model(sess)
            #sess.run(running_vars_initializer)
            prev_loss = 100
            print("---STARTED TRAINING---")
            for epoch in tqdm(range(self.epochs)):
                loss_epoch = 0 #Track loss after each epoch
                for b in tqdm(range(0,len(self.dataset)-self.batch_size_,self.batch_size_)):
                    batch_x = self.dataset[b:b+self.batch_size_]
                    _,cost = sess.run([appliedgrad,freecost],feed_dict={self.x:batch_x})
                    loss_epoch += abs(cost)
                    loss_list.append(loss_epoch/len(self.dataset))
                    epoch_list.append(epoch)
                print("\nLoss",loss_epoch/len(self.dataset),"at epoch",epoch)
                #if (loss_epoch/len(self.dataset))<prev_loss:
                saver.save(sess,"./TrainingData/STACKED-LSTM-RBM/"+"DEEPTIMESTEPS"+str(self.num_timesteps)+"epoch"+str(epoch)+"$"+str(loss_epoch/len(self.dataset))+".ckpt")
                    #prev_loss = loss_epoch/len(self.dataset)

            #score = sess.run(tf_metric)
            #print("[TF] SCORE: ", score)

        writer.close()
        plt.plot(epoch_list,loss_list)
        plt.title('LSTM-RBM Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def getRandomNotes(self):
        '''Generate n_visible number of notes'''
        a = np.zeros([1,self.n_visible])
        offset = 0
        for i in range(0,self.n_visible-(len(self.duration_dict)+len(self.pitch_oct_dict)),len(self.pitch_oct_dict)):
            a[0,i+np.random.randint(0,len(self.pitch_oct_dict))] = 1
        a[0,np.random.randint(0,len(self.duration_dict))+((len(self.pitch_oct_dict))*4)] = 1
        return a

    def test(self,training_weights):
        '''Tests the model trained saved at training_weights path (imporvises music)'''
        saver = tf.train.Saver([self.W,self.Wuh,self.Wuv,self.Wvu,self.Wuu,self.bh,self.bv,self.bu,self.u0,self.c0,self.u1,self.c1,self.u2,self.c2,self.W1,self.U1,self.b1,self.W2,self.U2,self.b2])

        #Random input for initialization of visible layer
        primer = self.getRandomNotes()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,training_weights)

            for i in tqdm(range(1)):
                generated_music = sess.run(self.generate(),feed_dict={self.x:primer})

            #array of array: 4 notes + time/duration
            chord_list = []
            for visible_layer in generated_music:
                chord = []
                offset = len(self.pitch_oct_dict)
                prior = 0
                for i in range(4): #For notes/pitches
                    chord.append(np.argmax(visible_layer[prior:offset]))
                    prior = offset
                    offset += len(self.pitch_oct_dict)
                #For duration
                prior = offset
                offset += len(self.duration_dict)
                chord.append(np.argmax(visible_layer[prior:offset]))
                #print(len(visible_layer[prior:offset])==len(self.duration_dict))
                chord_list.append(chord)
            self.create_midi(chord_list)

    def create_midi(self,prediction_output):

        inv_duration = {duration: dnum for dnum,duration in self.duration_dict.items()}
        inv_pitch_oct =  {p_o: enc for enc,p_o in self.pitch_oct_dict.items()}

        import datetime
        fmt = '%Y%m%d%H%M%S'
        now_str = datetime.datetime.now().strftime(fmt)
        print(prediction_output)
        '''create usinf midifile'''
        mf = MIDIFile(4) #4 parts
        time = 0
        channel = 1
        volume = 100

        for a_chord in prediction_output:
            d = float(inv_duration[a_chord[4]])
            for i in range(4):
                p = int(inv_pitch_oct[a_chord[i]])
                mf.addNote(i,channel,p,time,d,volume)
            time += d

        with open("./GeneratedMusic/DEEP_LSTM_SONG_MIDI"+now_str+".mid","wb") as outf:
            mf.writeFile(outf)

        '''
        create using music21
        dirstr ="./GeneratedMusic/DEEP_LSTM_RBM_SONG"+now_str+".mid"
        song = stream.Stream()

        for a_chord in prediction_output:
            d = inv_duration[a_chord[4]]
            gen_chord = []
            for i in range(4): #each pitch/octave encoding in chord
                encoding = inv_pitch_oct[a_chord[i]].split(';')
                p = inv_pitch[int(encoding[0])]
                o = inv_octave[int(encoding[1])]

                a_note = note.Note(str(p))
                a_note.octave = o
                a_note.duration.quarterLength = d
                gen_chord.append(a_note)

            song.append(chord.Chord(gen_chord))
        song.write('mid',fp=dirstr)
        '''

    def generate(self):
        '''Generates music by propagating through the LSTM and sampling from the RBM '''
        lstm_state = tf.scan(self.deep_step, self.x, initializer=[self.u0,self.c0,self.u1,self.c1,self.u2,self.c2])
        Uarr = lstm_state[4]

        U = Uarr[int(np.floor((self.num_timesteps-1)/self.num_timesteps)), :, :]
        ts = tf.TensorShape
        [_, _, _, _, music] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                self.generate_recurrence, [tf.constant(1), tf.constant(self.num_timesteps*100), [lstm_state[0],lstm_state[1],lstm_state[2],lstm_state[3],U,lstm_state[5]],
                                                tf.zeros([1, self.n_visible], tf.float32),tf.zeros([1, self.n_visible],  tf.float32)],
                                                shape_invariants=[ts([]), ts([]), [lstm_state[0].get_shape(),lstm_state[1].get_shape(),lstm_state[2].get_shape(),lstm_state[3].get_shape(),U.get_shape(),lstm_state[5].get_shape()], ts([1, self.n_visible]),  ts([None, self.n_visible])])
        return music

    def generate_recurrence(self,count,k,prev_t,primer,music):
        #This function builds and runs the gibbs steps for each RBM in the chain to generate music
        #Get the bias vectors from the current state of the RNN
        st_1 = prev_t[4]

        self.bv_t = tf.add(self.bv, tf.matmul(st_1, self.Wuv))
        self.bh_t = tf.add(self.bh, tf.matmul(st_1, self.Wuh))

        #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
        x_out = self.gibbs_sample(primer, k=25)
        '''INPUT LAYER- RBM VISIBLE LAYER Wvu ---> LSTM UNIT Wuu'''
        #xt = tf.reshape(xt,[1,self.n_visible])
        #Two states in LSTM internal cell state(ct) and external state/output(st)
        #get previous states
        st_1,ct_1= prev_t[0][count],prev_t[1]

        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(x_out,self.Wvu[0])+tf.matmul(st_1,self.Wuu[0])+self.bu[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(x_out,self.Wvu[1])+tf.matmul(st_1,self.Wuu[1])+self.bu[1])
        #output layer
        o = tf.sigmoid(tf.matmul(x_out,self.Wvu[2])+tf.matmul(st_1,self.Wuu[2])+self.bu[2])
        #some layer
        g = tf.tanh(tf.matmul(x_out,self.Wvu[3])+tf.matmul(st_1,self.Wuu[3])+self.bu[3])
        #update internal cell state
        ct = (ct_1*f)+(self.dropout(g)*i)
        #update external state
        st = tf.tanh(ct)*o

        '''HIDDEN LAYER 1'''
        sth_1,cth_1 = prev_t[2][count],prev_t[3]
        inp = st[count]
        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(inp,self.W1[0])+tf.matmul(sth_1,self.U1[0])+self.b1[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(inp,self.W1[1])+tf.matmul(sth_1,self.U1[1])+self.b1[1])
        #output layer
        o = tf.sigmoid(tf.matmul(inp,self.W1[2])+tf.matmul(sth_1,self.U1[2])+self.b1[2])
        #some layer
        g = tf.tanh(tf.matmul(inp,self.W1[3])+tf.matmul(sth_1,self.U1[3])+self.b1[3])
        #update internal cell state
        cth = (cth_1*f)+(self.dropout(g)*i)
        #update external state
        sth = tf.tanh(cth)*o

        '''OUTPUT LAYER: LSTM --> RBM (bht/bvt)'''
        sth_1,cth_1 = prev_t[4],prev_t[5]
        inp = sth[count]
        #Input layer:decides if new information is relevant then lets it in
        i = tf.sigmoid(tf.matmul(inp,self.W2[0])+tf.matmul(sth_1,self.U2[0])+self.b2[0])
        #forget layer:gets rid of irrelevant information
        f = tf.sigmoid(tf.matmul(inp,self.W2[1])+tf.matmul(sth_1,self.U2[1])+self.b2[1])
        #output layer
        o = tf.sigmoid(tf.matmul(inp,self.W2[2])+tf.matmul(sth_1,self.U2[2])+self.b2[2])
        #some layer
        g = tf.tanh(tf.matmul(inp,self.W2[3])+tf.matmul(sth_1,self.U2[3])+self.b2[3])
        #update internal cell state
        cth2 = (cth_1*f)+(self.dropout(g)*i)
        #update external state
        sth2 = tf.tanh(cth)*o

        #Add the new output to the musical piece
        music = tf.concat([music, x_out],0)

        return count+1, k,[st,ct,sth,cth,prev_t[4],cth2], x_out, music

    def compose(self):

        def compose_(i,k,prev_t,primer,pred):

            st_1, ct_1 = prev_t[0],prev_t[1]

            bv_t = tf.add(bv,tf.matmul(st_1,self.Wuv))
            bh_t = tf.add(bh,tf.matmul(st_1,self.Wuh))

            x_out =gibbs_sample(primer,self.W,self.bv_t,self.bh_t,k=25)

            #Propagate through the LSTM using the current output 'x_out' and the LSTM hidden unit at t-1, st_1, ct_1

            state = self.deep_step(prev_t,x_out)

            #Append x_out to prediction
            pred = tf.concat(values=[pred,x_out],axis=0)

            return i+1,k,state,x_out,x,pred

        lstm_state = tf.scan(self.deep_step,self.x,initializer=[self.u0,self.c0,self.u1,self.c1,self.u2,self.c2])

        s_t,c_t = lstm_state[4],lstm_state[5]

        pred = tf.zeros([1,n_visible],tf.float32)

        ts = tf.TensorShape


        #Repeat compose_ whilst i<n is True
        ts = tf.TensorShape  # To quickly define a TensorShape
        compose_loop_out = tf.while_loop(lambda i, n, *args: i < n, compose_, [tf.constant(1), tf.constant(song_timesteps), lstm_state,
                                         tf.zeros([1, self.n_visible], tf.float32), tf.zeros([1, self.n_visible], tf.float32)],
                                         shape_invariants=[ts([]), ts([]), [lstm_state[0].get_shape(),lstm_state[1].get_shape(),lstm_state[2].get_shape(),lstm_state[3].get_shape(),s_t.get_shape(),c_t.get_shape()], ts([1, self.n_visible]), ts([1,self.n_visible])])
        pred = compose_loop_out[10]
        return pred

#Convert music_obj into input sequences for training(reshape into timesteps)
def get_input_sequences(dataset,num_timesteps,n_visible):
    visible_layer_inputs = []
    for song in dataset: #Traverse each song in set
        for chord_index in range(0,len(song)-num_timesteps): #get timesteps of chords in song
            chord_set = song[chord_index:chord_index+num_timesteps]
            chord_encoding = np.zeros(n_visible) #initialize 0xn_visible
            offset = 0
            for chord in chord_set:
                for note_index in range(4):
                    chord_encoding[chord[note_index]+offset] = 1
                    offset += len(pitch_oct_dict)-1
                chord_encoding[chord[4]+offset] = 1
                offset += len(duration_dict)
            visible_layer_inputs.append(chord_encoding)
    #Clear unreferenced memory
    gc.collect()
    return visible_layer_inputs


if __name__ == '__main__':
    #Get music object(a set of songs represented in numeric format)
    '''
    dataset = ud.loadobj('./Files/BachChords')
    pitch_oct_dict = ud.loadobj('./Files/BachPitchOctave')
    duration_dict = ud.loadobj('./Files/BachDuration')
    octave_dict = ud.loadobj('./Files/BachOctaves')
    pitch_dict = ud.loadobj('./Files/BachPitch')
    '''
    dataset = ud.loadobj('./Files/BachMidiChords')
    pitch_oct_dict = ud.loadobj('./Files/BachMidiPitchDict')
    duration_dict = ud.loadobj('./Files/BachMidiDurationDict')
    octave_dict = None
    pitch_dict =None
    num_timesteps = 4
    n_visible = ((len(pitch_oct_dict)*4)+len(duration_dict))*num_timesteps
    n_hidden = int(n_visible*0.60)
    epochs = 1
    batch_size = 500
    input_sequences = get_input_sequences(dataset,num_timesteps,n_visible)
    model = lstm_rbm(input_sequences,n_visible=n_visible,n_hidden=n_hidden,epochs=epochs,batch_size=batch_size,num_timesteps=num_timesteps,pitch_dict=pitch_dict,duration_dict=duration_dict,pitch_oct_dict=pitch_oct_dict,octave_dict=octave_dict)
    model.train()
    #model.test("./TrainingData/STACKED-LSTM-RBM/DEEPTIMESTEPS2epoch308$0.045286486770334845.ckpt")
