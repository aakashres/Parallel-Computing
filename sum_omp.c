#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void main()
{
    omp_set_num_threads(8);
    int sum = 0;
    int lsum = 0;
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};

#pragma omp parallel private(lsum)
    {
        int i;
#pragma omp for
        for (i = 0; i < 8; i++)
        {
            lsum = lsum + A[i];
        }
#pragma omp critical
        {
            sum += lsum;
        }
    }
    printf("%d", sum);
}