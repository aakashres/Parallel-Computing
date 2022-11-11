/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < n && row < n)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int n = 10;
    int *h_a, *h_b, *h_c, *h_cc;
    int *d_a, *d_b, *d_c;
    
    /* Fixed seed for illustration */
    srand(3333);
    // allocate memory in host RAM, h_cc is used to store CPU result
    cudaMallocHost((void **)&h_a, sizeof(int) * n * n);
    cudaMallocHost((void **)&h_b, sizeof(int) * n * n);
    cudaMallocHost((void **)&h_c, sizeof(int) * n * n);
    cudaMallocHost((void **)&h_cc, sizeof(int) * n * n);

    // random initialize matrix A
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_b[i * n + j] = rand() % 1024;
        }
    }

    // Allocate memory space on the device
    cudaMalloc((void **)&d_a, sizeof(int) * n * n);
    cudaMalloc((void **)&d_b, sizeof(int) * n * n);
    cudaMalloc((void **)&d_c, sizeof(int) * n * n);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    // Launch kernel
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    // Transefr results from device to host
    cudaMemcpy(h_c, d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    cpu_matrix_mult(h_a, h_b, h_cc, n);
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if (h_cc[i * n + j] != h_c[i * n + j])
            {
                all_ok = 0;
            }
        }
        // printf("\n");
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%d ", h_c[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%d ", h_cc[i*n + j]);
        }
        printf("\n");
    }


    // roughly compute speedup
    if (all_ok)
    {
        printf("all results are correct!!!");
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}