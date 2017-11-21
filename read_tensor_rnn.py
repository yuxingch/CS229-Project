from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class RnnModel:
    
    def __init__(self, placeholder, min_step=10, state_dim=128):
        self.min_step = min_step
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(state_dim)
        # Initial state of the LSTM memory.
        self.musicinput = placeholder['music_input']
        self.batch_size = tf.shape(self.musicinput)[0]
        self.range = tf.shape(self.musicinput)[1]
        self.input_dim = tf.shape(self.musicinput)[2]
        self.targets_pitch = self.musicinput[min_step:,:]
        self.initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        self.build()
        self.loss()
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss)

    def build(self):

        # The value of state is updated after processing each batch of notes.
        # The LSTM output can be used to make next pitch predictions
        outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.musicinput,
                                       initial_state=self.initial_state,
                                       dtype=tf.float32)
        

        W = tf.get_variable(name='weight',shape=(self.state_dim,128),dtype=tf.float32)
        b = tf.get_variable(name='bias',shape=128,dtype=tf.float32)
        logit = tf.matmul(outputs, W) + b 
        prob= tf.nn.softmax(logit) 
        self.pred = prob[:-min_step, :]


    def loss(self):
        target_flattened = tf.reshape(self.target_pitch, 
                                      [self.batch_size, self.range*self.input_size])
        pred_flattened = tf.reshape(self.pred,
                                    [self.batch_size, self.range*self.input_size])
        # compute cost
        loss = tf.losses.softmax_cross_entropy(target_pitch, logits=self.outputs[:-1,:]) 
        self.loss = tf.sum(loss)


    


