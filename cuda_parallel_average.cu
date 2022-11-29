/*
To compile:
module load gcc8/8.4.0
module load nvidia_hpcsdk
nvcc cuda_parallel_average.cu -o cuda_parallel_average

TO execute: Create a slurm script and submit
#!/bin/bash
#SBATCH -J KERNELS # job name
#SBATCH -o class_%j.txt # output and error file name %j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p classroomgpu # queue (partition) -- defq, eduq, gpuq, short
#SBATCH -t 00:02:00 # run time (hh:mm:ss) - 2.0 mins in this example.
#SBATCH --gres=gpu:1

module load gcc8/8.4.0
module load slurm
module load nvidia_hpcsdk

./cuda_parallel_average

*/

#include <stdlib.h>
#include <stdio.h>
#include  <math.h>
#define INPUT_SIZE 1000

__global__ void sum_reduction(int *v, int *v_r)
{
    __shared__ int partial_sum[INPUT_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_sum[tid] = v[tid];
    __syncthreads();
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid; 
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index+s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[tid];
    }
}

int main()
{
    size_t bytes = INPUT_SIZE * sizeof(int);
    int *h_input, *h_output;
    int *d_input, *d_output;
    h_input = (int *)malloc(bytes);
    h_output = (int *)malloc(bytes);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    for (int i = 0; i < INPUT_SIZE; i++)
        h_input[i] = i+1;
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    int TB_SIZE =1000;
    int GRID_SIZE =  (int) INPUT_SIZE / TB_SIZE;
    sum_reduction<<<GRID_SIZE, TB_SIZE>>>(d_input, d_output);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("Average of array is : %0.3f\n", (double)h_output[0] / INPUT_SIZE);
    return 0;
}