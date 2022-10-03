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
    omp_set_num_threads(5);
    int i = 100;
#pragma omp parallel
    {
#pragma omp critical
        {
            i++;
        }
        printf("%d\n", i);
    }
    printf("%d\n", i);
    return 0;
}