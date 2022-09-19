#include <stdio.h>
#include <omp.h>

#define INPUT_SIZE 1000
#define NUM_THREADS 5

/**
 * @brief Program to calculate average of INPUT_SIZE array containing 1 to INPUT_SIZE in parallel manner using OpenMP
 *
 *  INPUT_SIZE: Total size of the array
 *  NUM_THREADS: Total threads running in parallel to calculate partial sum of the array
 *
 *
 * @return int execution status of the main program
 */
int main()
{
    {
        omp_set_num_threads(NUM_THREADS);
        double average = 0;
        int sum = 0;

        // Create an input array of length INPUT_SIZE containing 1 to INPUT_SIZE integers
        // Using Prallel threads to fill in the array as  well
        int input[INPUT_SIZE];
        {
#pragma omp parallel for
            for (int i = 0; i < INPUT_SIZE; i++)
                input[i] = i + 1;
        }
        // Parallelizing average using Openmp
        // Parallel for loop to calculate sum of number divided into NUM_THREADS
        // I used critical section previously to sum all the thread_sums into one value
        // Since, this is an reduction step. This can easily be reduced using reduction declaritive which poses less overhead
        {
#pragma omp parallel for reduction(+ \
                                   : sum)
            for (int i = 0; i < INPUT_SIZE; i++)
                sum += input[i];
        }

        average = (double)sum / INPUT_SIZE;
        printf("Parallel Average = %0.2f\n", average);
    }
    return 0;
}