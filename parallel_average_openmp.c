#include <stdio.h>
#include <omp.h>

#define INPUT_SIZE 1000
#define NUM_THREADS 5

/**
 * @brief Program to calculate average of INPUT_SIZE array containing 1 to INPUT_SIZE in parallel manner using OpenMP
 *
 * @return int execution status
 */
int main()
{
    {
        omp_set_num_threads(NUM_THREADS);
        double average = 0;
        int thread_sum = 0;
        // Create an input array of length INPUT_SIZE containing 1 to INPUT_SIZE integers
        int input[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            input[i] = i + 1;
        }
        // Parallelizing average using Openmp
#pragma omp parallel private(thread_sum)
        {
            int i;
            // Parallel for loop to calculate sum of number divided into NUM_THREADS
#pragma omp for
            for (i = 0; i < INPUT_SIZE; i++)
            {
                thread_sum += input[i];
            }
            // summing the thread_sum into one single sum. Critical denotes only one thread can execute at once
#pragma omp critical
            {
                average += thread_sum;
            }
        }
        average = average / INPUT_SIZE;
        printf("Parallel Average = %0.2f\n", average);
    }
    return 0;
}