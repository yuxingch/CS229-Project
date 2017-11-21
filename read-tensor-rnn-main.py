from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

def tensorflow_music_input(music_input):
    
    t_music_input = tf.placeholder(music_input,shape=[None,music_input.shape[1],music_input.shape[2]],
                                   dtype=tf.float32,name='music_input')
    return {'music_input':t_music_input}


def matrix_to_input(channelMatrix):
    # note vector which indicate which notes are played in a certain time step
    note_start = 0
    note_end = 128
    note_size= note_end - note_start
    
    # total time step
    time_size = len(channelMatrix[0])
   
    # a vector of notes in a current time step 1*5
    curr_vector = np.zeros((5,),dtype=np.int32)
    music_input= np.zeros((note_size,time_size),dtype=np.float)
    
    # resize the note vector to 1*note_size
    for i in range(time_size):  
        for j in range(5):  
            curr_vector[j]= channelMatrix[j][i].astype(np.int32)
            if(curr_vector[j]==0):
                continue
            music_input[curr_vector[j]-note_start+1][i] = 1.0
            
    #print(curr_vector)
            
    return music_input
    
    
def main():  
    # channel 0
    channelMatrix = np.load('channel0.npy')
    music_input = matrix_to_input(channelMatrix)
    #print(len(music_input[:,-1]))
    
    
    
if __name__ == "__main__":
    main()