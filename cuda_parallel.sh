#!/bin/bash
#SBATCH -J KERNELS # job name 
#SBATCH -o parallel_%j.txt # output and error file name %j expands to jobID) 
#SBATCH -n 1 # total number of tasks requested 
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p classroomgpu # queue (partition) -- defq, eduq, gpuq, short 
#SBATCH -t 00:02:00 # run time (hh:mm:ss) - 2.0 mins in this example. 
#SBATCH --gres=gpu:1

module load gcc8/8.4.0
module load slurm
module load nvidia_hpcsdk


./bin/parallel_cuda