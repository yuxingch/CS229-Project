from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class RnnModel:
    
    def __init__(self, placeholder, min_step=10, state_dim=128):
        self.min_step = min_step
        self.state_dim = state_dim
        self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(state_dim)
        # Initial state of the LSTM memory.
        self.music_input = placeholder['music_input']
        self.batch_size = tf.shape(self.music_input)[0]
        self.range = tf.shape(self.music_input)[1]
        self.input_dim = tf.shape(self.music_input)[2]
        self.targets_pitch = self.music_input[min_step:,:]
        self.initial_state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.build()
        self.loss()
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

    def build(self):

        # The value of state is updated after processing each batch of notes.
        # The LSTM output can be used to make next pitch predictions
        outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.music_input,
                                       initial_state=self.initial_state,
                                       dtype=tf.float32)
        
        #output_trunc = outputs[:-min_step, :]
        #output_flattened = tf.reshape(output_truc, shape=[self.batch_size, 
        #                                                  tf.shape(output_trunc)[1] *
        #                                                  tf.shape(output_truct)[2]],
        #                              name='flatten_output')

        #W = tf.get_variable(name='weight',shape=(,128),dtype=tf.float32)
        #b = tf.get_variable(name='bias',shape=128,dtype=tf.float32)
        #logit = tf.matmul(outputs, W) + b 
        prob= tf.nn.softmax(outputs) 
        self.pred = prob[:, :-self.min_step, :]


    def loss(self):
        target_flattened = tf.reshape(self.targets_pitch, 
                                      [self.batch_size, self.range*self.input_dim], 'reshape_target')
        pred_flattened = tf.reshape(self.pred,
                                    [self.batch_size, self.range*self.input_dim], 'reshape_pred')
        # compute cost
        
        loss = tf.losses.softmax_cross_entropy(target_flattened, logits=pred_flattened) 
        self.loss = tf.reduce_sum(loss)
        tf.summary.scalar('loss', self.loss)



    

