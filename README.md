# NanosigSim
Simulation of Nanopore Sequencing Signals Based on BiGRU

# Overview
Oxford Nanopore sequencing is an important sequencing technology, which reads the nucleotide sequence by detecting the electrical current signal changes when DNA molecule is forced to pass through a biological nanopore. The research on signal simulation of nanopore sequencing is highly desirable for method developments of nanopore sequencing applications. To improve the simulation accuracy, we propose a novel signal simulation method based on Bi-directional Gated Recurrent Units (BiGRU). We named the proposed method NanosigSim.

# Download the NanosigSim package
`git clone https://github.com/zpllx/NanosigSim.git`  
`cd NanosigSim`

# Usage
## Simulate the signal for a given sequence
`python signal_simulation.py -i fasta/1.fasta -m model -o simulation` 
## Simulate the signals for multiple given sequences
`python signal_simulation.py -p fasta -m model -o simulation` 
## Train the signal processing model based on BiGRU
`python create_train_tfrecord.py -p train_data -o tfrecords`   
`python train_simulation.py -p tfrecords -o model` 


# Simulated Signal VS Real Raw Signal
## Real Raw Signal
![image](https://github.com/zpllx/NanosigSim/blob/main/simulation/example/Real%20raw%20signal.jpeg)
## Simulated Signal


