#include <stdio.h>
#include <omp.h>

long nfib(long n)
{
    long i, j;
    if (n < 2)
    {
        return 1;
    }
#pragma omp task shared(i) if (n > 33)
    i = nfib(n - 1);
#pragma omp task shared(j) if (n > 33)
    j = nfib(n - 2);
#pragma omp taskwait
    return (i + j + 1);
}

int main()
{
    omp_set_num_threads(10);
    long v;
    long n = 100;
#pragma omp parallel shared(n, v)
#pragma omp single
    {
        v = nfib(n);
    }
    printf("%d", v);
    return 0;
}
