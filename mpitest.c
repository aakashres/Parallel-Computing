#include <stdio.h>
#include <mpi.h>

#define INPUT_SIZE 1000

/* 
    To load necessary modules:
        module load mpich/ge/gcc/64/3.2.1
        module load gcc8/8.4.0
    To compile: 
        mpicc -o mpi_parallel parallel_average_mpi.c
    To execute:
        mpirun -np <number_of_process> ./mpi_parallel
*/
int main(int argc, char *argv[])
{
    int i, size, rank;
    MPI_Status Stat;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char outmsg = 'a' , inmdg = 'b';
    int dest, source;
    int tag = 1;
    if (rank==0) {
        dest = 1;
        source = 1;
        MPI_Send(&outmsg,1,MPI_CHAR,dest,tag,MPI_COMM_WORLD);
        MPI_Recv(&inmdg,1,MPI_CHAR,source,tag,MPI_COMM_WORLD,&Stat);
        printf("%c", inmdg);
    }
    else if (rank == 1) {
        dest = 0;
        source = 0;
        MPI_Recv(&inmdg,1,MPI_CHAR,source,tag,MPI_COMM_WORLD,&Stat);
        printf("%c", inmdg);
        MPI_Send(&outmsg,1,MPI_CHAR,dest,tag,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
