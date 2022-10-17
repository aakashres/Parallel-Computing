#!/bin/bash
#SBATCH -J CompileKernels # job name 
#SBATCH -o ./r2-logs/log_out_%j.o # output and error file name %j expands to jobID) 
#SBATCH -n 1 # total number of tasks requested 
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p shortq# queue (partition) -- defq, eduq, gpuq, short 
#SBATCH -t 00:02:00 # run time (hh:mm:ss) - 2.0 mins in this example. 

module load gcc8/8.4.0
module load mpich/ge/gcc/64/3.2.1
module load slurm

make mpi_parallel
mpirun -np 1 ./bin/mpi_parallel
make clean
