from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from read_tensor_rnn import RnnModel
import numpy as np
import tensorflow as tf
import math


def tensorflow_music_input(music_input):
    
    t_music_input = tf.placeholder(tf.float32,shape=[None,music_input.shape[1],music_input.shape[2]],
                                   name='music_input')
    return {'music_input':t_music_input}


def matrix_to_input(channelMatrix):
    # note vector which indicate which notes are played in a certain time step
    note_start = 0
    note_end = 128
    note_size= note_end - note_start
    
    # total time step
   
    # TODO: fix batch size
    batch_size=1
    music_input= np.zeros((batch_size,note_size,time_size),dtype=np.float)
    
    # resize the note vector to 1*note_size
    for m in range(batch_size):
        for i in range(time_size):  
            for j in range(5):  
                curr_vector[j]= channelMatrix[m][j][i].astype(np.int32)
                if(curr_vector[j]==0):
                    continue
                music_input[m][curr_vector[j]-note_start+1][i] = 1.0
            
    #print(curr_vector)
            
    return music_input
    
    
def main(argv=None):  
    # channel 0
    channelMatrix = np.load('channel0.npy')
    channelMatrix = np.expand_dims(channelMatrix , axis=0)
    print(channelMatrix.shape)
    music_input = matrix_to_input(channelMatrix)
    music_input = np.transpose(music_input)
    print(music_input.shape)
    
    placeholder = tensorflow_music_input(music_input)
    model = RnnModel(placeholder)
    
    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    _, loss = sess.run([model.train_op, model.loss], feed_dict={placeholder['music_input']: music_input})
    
    print(loss)
    
    
if __name__ == "__main__":
    tf.app.run()
