#!/bin/bash
# not on cluster GPU, just CPU
module load python/3.7.2
module load tensorflow/1.12.0-py37-gpu
module load keras/2.2.5-py37

# for cluster GPU
module load python/3.6.1
module load tensorflow/1.12.0-py36-gpu
module load keras/2.2.5-py36
