import tensorflow as tf
import argparse
import os
from   time import strftime, localtime
import sys
import random

# get time
def  get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def  string_to_int(label,char_map_dict=None):
    int_list = []
    for c in label:
        int_list.append(char_map_dict[c])
    return int_list

def  int64_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def  bytes_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def  float_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# get files to be processed
def  parse_files(args):
    if args.path:
        if not os.path.isdir(args.path):
            print('Provided [train-basedir] is not a directory.')
            exit()
        train_basedir = (args.path if args.path.endswith('/') else args.path + '/')
        all_train_files = []
        for root, _, fns in os.walk(train_basedir):
            for fn in fns:
                if not fn.endswith('.train'):
                    continue
                all_train_files.append(os.path.join(root, fn))
        if len(all_train_files) < 1:
            print('No files identified in the specified directory or within immediate subdirectories.')
            exit()
        print('[%s] There are %d train files to be processed!' % (get_time(),len(all_train_files)),end='')
        return all_train_files

def  write_tfrecord(train_files,args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    tfrecords_path = os.path.join(args.output,'train.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i, train_file in enumerate(train_files):
            
            file = open(train_file,'r+')
            seq_signal = file.readline()
            seq_signal = seq_signal.split()
            seq_signal = [float(x) for x in seq_signal]
            raw_signal = file.readline()
            raw_signal = raw_signal.split()
            raw_signal = [float(x) for x in raw_signal]
            file.close()
            
#            raw_signal = np.array(raw_signal,dtype=np.float32).reshape((-1,1)).tolist()
            seq_signal = seq_signal
            raw_signal = raw_signal
            sample_name = train_file[:-6].encode('utf-8')
            
            features = tf.train.Features(feature={
                    'seq_signals': float_feature(seq_signal),
                    'raw_signals': float_feature(raw_signal),
                    'sample_name': bytes_feature(sample_name)
                    })
    
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r[%s]>> Writing to train.tfrecords %d / %d' % (get_time(),i+1,len(train_files)))
            sys.stdout.flush()
        sys.stdout.write('\n[%s]>> train.tfrecords write finish.' % (get_time()))
        sys.stdout.flush()
            

def  convert_dataset(train_files,args):   
    random.shuffle(train_files)
    write_tfrecord(train_files,args)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Nanopore Sequence train data')
    parser.add_argument('-p', '--path', type=str, help= 'Top directory path of train files')
    parser.add_argument('-o', '--output', type=str, default="tfrecords/", help= 'Directory where tfrecords are written to')
    args = parser.parse_args()
    
    train_files = parse_files(args)
    sys.stdout.write('\n[%s] Begin to convert dataset!\n' % (get_time()))
    sys.stdout.flush()
    convert_dataset(train_files,args)
    sys.stdout.write('\n[%s] End!' % (get_time()))
    sys.stdout.flush()
