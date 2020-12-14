import  os
import  argparse 
import  sys
import  tensorflow as tf
from    time import strftime, localtime
from    simulation_network  import  rawsignal_simulation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get time
def  get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

# get files to be processed
def  parse_files(args):
    if args.path:
        if not os.path.isdir(args.path):
            print('Provided [fasta-basedir] is not a directory.')
            exit()
        fastas_basedir = (args.path if args.path.endswith('/') else args.path + '/')
        all_fastas = []
        for root, _, fns in os.walk(fastas_basedir):
            for fn in fns:
                if not fn.endswith('.fasta'):
                    continue
                all_fastas.append(os.path.join(root, fn))
        if len(all_fastas) < 1:
            print('No files identified in the specified directory or within immediate subdirectories.')
            exit()
        print('[%s] There are %d fasta files to be processed!' % (get_time(),len(all_fastas)))
        return all_fastas
    else:
        all_fastas = []
        if not args.ind.endswith('.fasta'):
            print('Provided file is not a fasta_file.')
            exit()
        all_fastas.append(args.ind)
        print('There are %d fasta files to be processed!' % (len(all_fastas)))
        return all_fastas

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'Nanopore Sequence Signal Simulation Software')
    parser.add_argument('-p', '--path', type=str, help= 'Top directory path of fasta files')
    parser.add_argument('-i', '--ind', type=str, help= 'Individual fasta file')
    parser.add_argument('-m', '--model', type=str, default='model/', help= 'Base directory for the model')
    parser.add_argument('-o', '--output', type=str, default="simulation/", help= 'The raw signal simulation result')
    args = parser.parse_args()
    
    if (not args.path) and (not args.ind):
        print('Nothing to simulation!')
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    os.makedirs(args.output + '//raw_signal')
    os.makedirs(args.output + '//fast5')
    # get files to be processed
    files = parse_files(args)
    rawsignal_simulation(files,args)
    sys.stdout.write('\n[%s] The raw signal simulation is END!\n' % (get_time()))
    sys.stdout.flush()
