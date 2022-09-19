#include <stdio.h>
#include <omp.h>

#define INPUT_SIZE 1000
#define NUM_THREADS 5

/**
 *  If no scope is defined, OMP only parallize first statement after the pragma
 *
 *
 * @return int execution status of the main program
 */
int main()
{
    int id, numThreads;
    omp_set_num_threads(10);
#pragma omp parallel private(id, numThreads)
    {
        id = omp_get_thread_num();
        numThreads = omp_get_num_threads();
        printf("I am thread %d out of %d threads \n", id, numThreads);
    }
    return 0;
}