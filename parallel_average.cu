/*
To compile:
module load gcc8/8.4.0
module load nvidia_hpcsdk
nvcc parallel_average.cu -o parallel_average

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

./parallel_average

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 1000

// CUDA kernel. Each thread takes care of one element of c
__global__ void arrAdd(int *input, int *output, int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[])
{
    // Host input vectors
    int *h_input;
    int *h_output;
    // Device input vectors
    int *d_input;
    int *d_output;

    // Size, in bytes, of each vector
    size_t bytes = INPUT_SIZE * sizeof(int);

    int blockSize, gridSize;
    // Number of threads in each thread block
    blockSize = 250;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)INPUT_SIZE / blockSize);

    // Allocate memory for each vector on host
    h_input = (int *)malloc(bytes);
    h_output = (int *)malloc(bytes);

    for (int i = 0; i < INPUT_SIZE; i++)
        h_input[i] = i + 1;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy host vectors to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Execute the kernel
    arrAdd<<<gridSize, blockSize>>>(d_input, d_output, INPUT_SIZE);

    // Copy array back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within error
    // double sum = 0;
    // for(int i=0; i<gridSize; i++)
    //     sum += h_output[i * blockSize];
    // printf("final result: %f\n", sum/(double)INPUT_SIZE);

    // for (int i = 0; i < INPUT_SIZE; i++)
        // printf("final result: %d\n", h_output[i]);

    // Release device memory
    cudaFree(d_input);

    // Release host memory
    free(h_input);
    free(h_output);

    return 0;
}