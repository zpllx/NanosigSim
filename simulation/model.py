# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:58:31 2020

@author: ZhangPeng
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RNNNetwork(object):
    
    def __init__(self, hidden_num, layers_num):
        self.__hidden_num = hidden_num
        self.__layers_num = layers_num
        return
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def __signal_simulation(self, input_tensor, input_sequence_length):
        with tf.variable_scope('GRU_Layers'):
#            # forward lstm cell
#            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
#            # Backward direction cells
#            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_num]*self.__layers_num]
#            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
#                fw_cell_list, bw_cell_list, input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)
            
            
            # forward lstm cells
            cell_fw1 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_fw_1')
            cell_fw2 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_fw_2')
            cell_fw3 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_fw_3')
            cells_fw = [cell_fw1,cell_fw2,cell_fw3]            
#            cells_fw = [tf.nn.rnn_cell.BasicLSTMCell(unit) for unit in [self.__hidden_num]*self.__layers_num] 
            
            # backward latm cells
            cell_bw1 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_bw_1')
            cell_bw2 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_bw_2')
            cell_bw3 = tf.nn.rnn_cell.GRUCell(self.__hidden_num,name='cell_bw_3')
            cells_bw = [cell_bw1,cell_bw2,cell_bw3]           
#            cells_bw = [tf.nn.rnn_cell.BasicLSTMCell(unit) for unit in [self.__hidden_num]*self.__layers_num] 

            stack_lstm_layer, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=cells_fw, cells_bw=cells_bw, inputs=input_tensor, sequence_length=input_sequence_length, dtype=tf.float32)

            
            [batch_size, _, output_hidden_num] = stack_lstm_layer.get_shape().as_list()
            [batch_size, _, hidden_num] = input_tensor.get_shape().as_list()
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, output_hidden_num])

            # Doing the affine projection
            w1 = tf.Variable(tf.truncated_normal([output_hidden_num, 1], stddev=0.01), name="w1")
            b1 = tf.Variable(tf.constant(0.1,shape=[1]), name='b1')
            logits = tf.matmul(rnn_reshaped, w1) + b1
            logits = tf.reshape(logits, [batch_size, -1, 1])
            # Swap batch and batch axis
#            logits = tf.nn.softmax(logits)
            rnn_out = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        return rnn_out,logits

    def build_network(self, input_signal, sequence_length=None):
        # second apply the sequence label stage
        net_out,logits = self.__signal_simulation(input_tensor=input_signal, input_sequence_length=sequence_length)
        return net_out,logits