#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define INPUT_SIZE 1000
#define NUM_THREADS 5

struct sum_parameters
{
    int *array;
    int start_index;
    int end_index;
};

/**
 * @brief Computes the sum of array from [start_index, end_index)
 *
 * @param args_ptr
 * @return void*
 */
void *sum_array(void *args_ptr)
{
    struct sum_parameters params = *((struct sum_parameters *)args_ptr);
    int *array = params.array;
    int start_index = params.start_index;
    int end_index = params.end_index;
    size_t sum = 0;
    for (int i = start_index; i < end_index; i++)
    {
        sum += array[i];
    }
    return (void *)sum;
}

/**
 * @brief Calculates Serial Sum
 *
 * @param input_arr array whose average is to be calculated
 * @return double average of the array
 */
double average_serial(int *input_arr)
{
    struct sum_parameters param;
    param.array = input_arr;
    param.start_index = 0;
    param.end_index = INPUT_SIZE;
    size_t sum = (size_t)sum_array(&param);
    return (double)sum / INPUT_SIZE;
}

/**
 * @brief Calculates Parallel Sum
 * Divides array chunk into NUM_THREADS
 * Spawns NUM_THREADS - 1 thread and calculates sum of NUM_THREADS - 1 chunk
 * Main thread is considred to calculate sum for last chunk
 *
 * @param input_arr array whose average is to be calculated
 * @return double average of the array
 */
double average_parallel(int *input_arr)
{
    pthread_t threads[NUM_THREADS - 1];
    struct sum_parameters params[NUM_THREADS];
    const int chunk_size = INPUT_SIZE / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        params[i].array = input_arr;
        params[i].start_index = i * chunk_size;
        params[i].end_index = (i + 1) * chunk_size;
    }
    params[NUM_THREADS - 1].end_index = INPUT_SIZE;
    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        pthread_create(&threads[i], NULL, sum_array, (void *)&params[i]);
    }
    size_t sum = (size_t)sum_array(&params[NUM_THREADS - 1]);
    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        size_t result;
        pthread_join(threads[i], (void *)&result);
        sum += result;
    }
    return (double)sum / INPUT_SIZE;
}

/**
 * @brief
 * Main for calculating average of N numbers using
 *  1. Serial Execution
 *  2. Parallel Execution
 *
 * To compile:
 *  gcc -o average -lpthread average.c
 *
 * To execute:
 *  ./average <serial|parallel>
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Insufficient Argiment: Usage %s <serial/parallel> \n", argv[0]);
        exit(1);
    }
    double average;
    int input[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        input[i] = i + 1;
    }
    if (strcmp(argv[1], "serial") == 0)
    {
        average = average_serial(input);
        printf("Serial Average from 1 to %d = %0.3f\n", INPUT_SIZE, average);
        return 0;
    }
    else if (strcmp(argv[1], "parallel") == 0)
    {
        average = average_parallel(input);
        printf("Parallel Average from 1 to %d  = %0.3f\n", INPUT_SIZE, average);
        pthread_exit(NULL);
        return 0;
    }
    else
    {
        printf("Invalid Argument: Usage %s <serial/parallel> \n", argv[0]);
        return (-1);
    }
}