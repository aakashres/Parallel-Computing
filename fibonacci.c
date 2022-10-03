#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char *argv[])
{
    int nth_term = atoi(argv[1]);
    int return_value = fibonacci(nth_term);
    printf("Your %dth Fibonacci number:  %d\n", nth_term, return_value);
    return 0;
}

int fibonacci(int n)
{
   long i, j;
   if (n<2){
        return 1;
        }
   #pragma omp task shared(i) if (n>33)
   i=fibonacci(n-1);
   #pragma omp tash shared(j) if (n>33)
   j=fibonacci(n-2);
   #pragma omp taskwait
   return i+j;
}
