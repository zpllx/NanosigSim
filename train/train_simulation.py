import  os
import  argparse
import  time
import  tensorflow  as tf
import  model
import  numpy  as np
import  sys
from    time import strftime, localtime

# get time
def  get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Basic prameters
NUM_THREADS = 1
STEP_PER_EVAL = 1                                                           # The number of training steps to run between evaluations
STEP_PER_SAVE = 150                                                            # The number of training steps to run between save checkpoints

BATCH_SIZE = 15                                                               # The number of samples in each batch
MAX_TRAIN_STEPS = 100000                                                       # The number of maximum iteration steps for training
LEARNING_RATE = 0.001                                                          # The initial learning rate for training.
DECAY_STEPS = 1000                                                             # The learning rate decay steps for training
DECAY_RATE = 0.8                                                               # The learning rate decay rate for training

lstm_hidden_uints = 128
lstm_hidden_layers = 3

def  read_tfrecord(tfrecord_path=None,num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannot find tfrecord file in path: %s' % (tfrecord_path))
    
    filename_queue = tf.train.string_input_producer([tfrecord_path],num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)    
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'seq_signals': tf.VarLenFeature(tf.float32),
                                           'raw_signals': tf.VarLenFeature(tf.float32),
                                           'sample_name': tf.FixedLenFeature([], tf.string),
                                       })
#    raw_signals = np.array([features['raw_signals']],dtype=np.float32).reshape((-1,1)).tolist()
    seq_signals = tf.sparse_tensor_to_dense(features['seq_signals'], default_value=0)
    seq_signals = tf.cast(seq_signals,tf.float32)
    seq_signals = tf.reshape(seq_signals,[-1,1])
    raw_signals = tf.sparse_tensor_to_dense(features['raw_signals'], default_value=0)
    raw_signals = tf.cast(raw_signals,tf.float32)
    raw_signals = tf.reshape(raw_signals,[-1,1])
    sequence_length = tf.cast(tf.shape(raw_signals)[-2],tf.int32)
    sample_name = features['sample_name']
    
    return seq_signals,raw_signals,sequence_length,sample_name


def  train_simulation(args):
    
    tfrecord_path = os.path.join(args.path,'train.tfrecord')
    seq_signals,raw_signals,sequence_lengths,_ = read_tfrecord(tfrecord_path=tfrecord_path)

    # decode the training data from tfrecords
    batch_seq_signals, batch_raw_signals, batch_sequence_lengths = tf.train.batch(
        tensors=[seq_signals, raw_signals, sequence_lengths], batch_size=BATCH_SIZE, dynamic_pad=True,
        capacity=10000 + 2*BATCH_SIZE, num_threads=NUM_THREADS)
    
    
    input_seq_signals = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, 1],name='input_seq_signals')
    input_raw_signals = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, 1],name='input_raw_signals')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32,shape=[BATCH_SIZE],name='input_sequence_lengths')
    
    # initialise the net model
    rnn_net = model.RNNNetwork(hidden_num=lstm_hidden_uints,
                                    layers_num=lstm_hidden_layers)
    with tf.variable_scope('RNN'):
        net_out,logits = rnn_net.build_network(input_seq_signals,input_sequence_lengths)  
    
    # compute Cross entropy loss
    new_input_labels = tf.reshape(input_raw_signals,[-1,1])
    
    logits = tf.reshape(logits,[-1,1])
    
    cross_entropy = tf.reduce_mean(tf.pow(tf.subtract(logits, new_input_labels), 2.))
        
    global_step = tf.train.create_global_step()
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=cross_entropy, global_step=global_step)   
#
    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='cross_entropy', tensor=cross_entropy)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    merge_summary_op = tf.summary.merge_all()
    
    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'nano_simulation_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(args.output, model_name)
    
    restore_path = tf.train.latest_checkpoint(args.output)

    with tf.Session() as sess:
        print("begin!******************************************")
        summary_writer = tf.summary.FileWriter(args.output)
        summary_writer.add_graph(sess.graph)

        # init all variables
        # sess.run(init_op)
        saver.restore(sess=sess,save_path=restore_path)
        print("hahhahahhaahahahha!")
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(MAX_TRAIN_STEPS):
            sys.stdout.write('\n[%s]step:%d' % (get_time(),step))
            sys.stdout.flush()
            
            signals, lbls, seq_lens = sess.run([batch_seq_signals, batch_raw_signals, batch_sequence_lengths])
            
            _, cl, lr, summary = sess.run(
                [optimizer, cross_entropy, learning_rate, merge_summary_op],
                feed_dict = {input_seq_signals:signals, input_raw_signals:lbls, input_sequence_lengths:seq_lens})

            if (step + 1) % STEP_PER_SAVE == 0: 
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % STEP_PER_EVAL == 0:
                # calculate the precision
                accuracy = 0.1
                print('\rstep:{:d} learning_rate={:9f} cross_entropy={:9f} train_accuracy={:9f}'.format(
                    step + 1, lr, cl, accuracy))
            
        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)     

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Train Nano_simulation network')
    parser.add_argument('-p', '--path', type=str, help= 'Top directory path of train files')
    parser.add_argument('-o', '--output', type=str, default="model/", help= 'The base directory for the model')
    args = parser.parse_args()
    
    train_simulation(args)