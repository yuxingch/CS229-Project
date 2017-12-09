from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from rnn_model import RnnModel

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import math
from random import randint


def tensorflow_music_input(music_input, seq_len):
    
    t_music_input = tf.placeholder(tf.float32,shape=[None,seq_len,music_input.shape[2]],
                                   name='music_input')
    return {'music_input':t_music_input}


def matrix_to_input(channelMatrix):
    # note vector which indicate which notes are played in a certain time step
    note_start = 0
    note_end = 128
    note_size= note_end - note_start
    
    # total time step
    time_size = channelMatrix.shape[1]
    #print(time_size)
    curr_vector = np.zeros((5,),dtype=np.int32)
    #print(curr_vector)
   
    # TODO: fix batch size
    batch_size=channelMatrix.shape[0]
    music_input= np.zeros((batch_size,time_size,note_size),dtype=np.float)
    
    #print(music_input.shape)
    # resize the note vector to 1*note_size
    for m in range(batch_size):
        for i in range(time_size):  
            for j in range(5):  
                curr_vector[j]= channelMatrix[m][i][j].astype(np.int32)
                #print(channelMatrix[m][i][j])
                if(curr_vector[j]==0):
                    continue
                music_input[m][i][curr_vector[j]-note_start-1] = 1.0
    
    #print(curr_vector)
    #print(music_input[m][i])
           
    return music_input 
    
    
def main(argv=None):  
    # channel 0, timestep from 1:50
    originalMatrix = np.load('new_channel0.npy')
    channelMatrix = originalMatrix[:,10000:-10000,:]
    
    # tranpose the matrix, now its dimension is timestep*note_size
    #channelMatrix  = np.transpose(channelMatrix )
    #print(channelMatrix.shape)
    
    #expand dimension, now batch_size*time*note_size
    #channelMatrix = np.expand_dims(channelMatrix , axis=0)
    #print(channelMatrix.shape)
    
    #expand note_size from 5*1 to 128*1
    music_input = matrix_to_input(channelMatrix)
    #print(music_input.shape)
    #print((music_input.shape)[0])
    #print((music_input.shape)[1])
    #print((music_input.shape)[2])
    
    seq_len = 50
    placeholder = tensorflow_music_input(music_input, seq_len)
    model = RnnModel(placeholder) 
    config = tf.ConfigProto(log_device_placement=False)
    
    # start tensorflow session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    print( tf.trainable_variables())
    merged = tf.summary.merge_all()
    #summary_writer
    max_value = originalMatrix.shape[1] - originalMatrix.shape[1] % 5000 - 1
    '''
    for i in range(18):
        left = 5000*i
        right = 5000*(i+1)
        channelMatrix = originalMatrix[:,left:right,:]
        music_input = matrix_to_input(channelMatrix)
        _,_, loss,accuracy = sess.run([merged,model.train_op, model.loss, model.accuracy], feed_dict={placeholder['music_input']: music_input})
    
        print(loss)
        print(accuracy)
    '''
    num_epoch =10
    print(music_input.shape[2])
    for epoch in range(num_epoch):
        for iter in range(200000):
            i = randint(0, channelMatrix.shape[1] - seq_len)
            batch_input = music_input[:,i:i+seq_len,:]
            _,_, loss, grad, pred, input = sess.run([merged, model.train_op, model.loss, model.grad,
                    model.pred, model.music_input], feed_dict={placeholder['music_input']: batch_input})

            if iter % 1000 == 0:
                print('Iter ', iter, ': loss=', loss)
                curr_grad = LA.norm(grad)
                print(curr_grad)
                if curr_grad < 1e-6:
                    break
            
            #print(accuracy)
        print("Loss for epoch %d = %f" % (epoch,loss)) #use this if we wanna generate a plot of loss vs. epoch
    print("Done Training")
    '''
    ## try music generation:
    
    ## initialization :
    state_dim=128   
    batch_size = music_input.shape[0]
    time_range = music_input.shape[1]
    note_input_dim = music_input.shape[2]
    note_output_dim = music_input.shape[2]
    print()
    new_state_gen = np.zeros([1,2*state_dim])
    start_vec = tf.placeholder(tf.float32,[1,1,note_input_dim])
    in_state = tf.placeholder(tf.float32,[1,2*state_dim])
    output, new_state = tf.nn.dynamic_rnn(cell=lstm,inputs = start_vec,initial_state=in_state,dtype=tf.float32)
    output = tf.reshape(output,[1,state_dim])
    out_logits = tf.matmul(output,W)+ b
    
    seq = [6] #the initial sequence we feed the LSTM
    if len(seq) > 1:
        x_init = np.reshape([make_feature_vec(i) for i in seq[:-1]],[1,len(seq)-1,note_input_dim])
        new_state_gen = session.run(state,feed_dict = {x:x_init})
    start = np.reshape(make_feature_vec(seq[-1]),[1,1,note_input_dim])
    for i in range(20):
        new_out_logits, new_state_gen = session.run([out_logits,new_state], feed_dict={start_vec:start,in_state:new_state_gen})
        index = int(tf.argmax(new_out_logits, 1).eval())
        seq.append(index)
        start = np.reshape(make_feature_vec(index),[1,1,note_input_dim])

    current = seq
    print('Final: {}'.format(current))
    current_converted = [num_to_vec[current[i]] for i in range(len(current))]
    print('Final converted: {}'.format(current_converted))
    '''
    
if __name__ == "__main__":
    tf.app.run()
