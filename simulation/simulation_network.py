# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:58:31 2020

@author: ZhangPeng
"""
import tensorflow as tf
import model
import numpy as np
import os
from   time import strftime, localtime
import h5py
import sys
import scipy.stats as st
import uuid
from shutil import copyfile
import random

gru_hidden_uints = 128
gru_hidden_layers = 3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get time
def  get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def rep_rvs(size,a=0.1, more=1, seed=0):
    a = a*5
    array_1 = np.ones(int(size*(0.075-0.015*a))).astype(int)
    samples = st.alpha.rvs(3.3928495261646932+a,
        -7.6451557771999035+(2*a), 50.873948369526737,
        size=(size-int(size*(0.075-0.015*a))), random_state=seed).astype(int)
    samples = np.concatenate((samples, array_1), 0)
    samples[samples<0] = 0
    samples[samples>40] = 40
    if more == 1:
        np.random.seed(seed)
        addi = np.array(abs(np.random.normal(2,1,size))).astype(int)
        samples[samples<9] += addi[samples<9]
        np.random.shuffle(samples)
        samples[samples<9] += addi[samples<9]
    return samples

def repeat_n_time(a, result, more, seed=0):
    rep_times = rep_rvs(len(result), a, more, seed)
    out = list()
    ali = list()
    pos = 0
    for i in range(len(result)):
        k = rep_times[i]
        cur = [result[i]] * k
        out.extend(cur)
        for j in range(k):
            ali.append((pos,i))
            pos = pos + 1
    event_idx = np.repeat(np.arange(len(result)), rep_times)
    return out,ali,event_idx

def  get_truth_signal(file_name,kmer_model):
    file_1 = open(file_name,'r+')
    lines = file_1.readlines()
    file_1.close()
    sequence = lines[1].strip()
    expect_signal = []
    for j in range(len(sequence)-5):
        expect_signal.append(kmer_model[sequence[j:j+6]])
    # repeat n times
    expect_signal, final_ali, event_idx = repeat_n_time(0.1, expect_signal, 1, seed=0)
    # median normalization
    shift = np.median(expect_signal)
    scale = np.median(np.abs(expect_signal - shift))
    expect_signal = (expect_signal - shift) / scale
    return expect_signal,shift,scale

def mod_raw_signal(fast5_fn, data_in, uid):
    ##Open file
    fast5_data = h5py.File(fast5_fn, 'r+')

    #Get raw data
    rk = list(fast5_data["Raw/Reads"].keys())[0]  #rk = read4
    raw_dat   = fast5_data['/Raw/Reads/'][rk]
    raw_attrs = raw_dat.attrs
    del raw_dat['Signal']
    raw_dat.create_dataset('Signal',data=data_in, dtype='i2', compression='gzip', compression_opts=9)  #-> with compression
    raw_attrs['duration'] = len(data_in)
    raw_attrs['read_id'] = uid
    fast5_data.close()

def create_fast5(signal_for_mod,fast5_file):
    # The new read id and the file name
    uid = str(uuid.uuid4())
    copyfile("template.fast5",fast5_file)
    #-> modify signal data inside fast5
    mod_raw_signal(fast5_file, signal_for_mod, uid)


def  rawsignal_simulation(files,args):
    #　get the k_mer model
    kmer_model = {}
    kmer_model_file = open("k_mer.model","r+")
    lines = kmer_model_file.readlines()
    kmer_model_file.close()
    for i in range(1,len(lines)):
        line = lines[i].split()
        kmer_model[line[0]] = float(line[1])
    
    input_raw_signal = tf.placeholder(dtype=tf.float32, shape=[1, None, 1], name='input_ground_truth_signal')
    input_sequence_length = tf.placeholder(dtype=tf.int32, shape=[1], name='input_signal_length')

    # initialise the net model
    rnn_net = model.RNNNetwork(hidden_num=gru_hidden_uints,
                               layers_num=gru_hidden_layers)

    with tf.variable_scope('RNN'):
         net_out,logits = rnn_net.build_network(input_raw_signal,input_sequence_length)
        
    
    init = tf.global_variables_initializer()

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(args.model)

    with tf.Session() as sess:
        
        #restore all variables
        #sess.run(init)
        saver.restore(sess=sess,save_path=save_path)
        
        for i in range(len(files)):
            try:
                file_name = files[i]
                ground_truth_signal,shift,scale = get_truth_signal(file_name,kmer_model)
                seq_len = np.array([len(ground_truth_signal)],dtype=np.int32)
                ground_truth_signal = np.array(ground_truth_signal,dtype=np.float32).reshape((1,len(ground_truth_signal),1))
                logits_ = sess.run(logits, feed_dict={input_raw_signal:ground_truth_signal, input_sequence_length:seq_len})
                logits_ = logits_.reshape(-1,1)
                logits_ = logits_.reshape(-1)
                logits_ = list(logits_)
                simulation_signal_1 = []
                simulation_signal = []
                for l in logits_:
                    simulation_signal_1.append(l * float(scale) + float(shift))
                noise_std = st.burr12(13.361341574188225, 0.28250388140535976, 2.1698912802503574, 0.5792746324544114).rvs(1)[0] - 0.7
                noise = list(np.random.normal(0,noise_std,len(simulation_signal_1)))
                for l in range(len(noise)):
                    simulation_signal_1[l] += noise[l]
                for l in simulation_signal_1:
                    simulation_signal.append(int(l / 0.16995123028755188 - 15))
                # output the raw_signal to file
                num = os.path.basename(files[i])[:-6]
                raw_signal_file = args.output + "/raw_signal/" + num + ".rawsig"
                file1 = open(raw_signal_file,"w+")
                for s in simulation_signal:
                    file1.write(str(s) + '\n')
                file1.close()
                # output the fast5 file
                fast5_file = args.output + "/fast5/" + num + ".fast5"
                create_fast5(simulation_signal,fast5_file)
                sys.stdout.write('\r[%s] Complete Schedule: %d / %d' % (get_time(),i+1,len(files)))
                sys.stdout.flush()
            except:
                pass
