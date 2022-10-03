#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    double starttime, endtime;
    MPI_Init(&argc, &argv);
    starttime = MPI_Wtime();
    int world_rank;
    int world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //pass the token

    int token =1000;

    for (int i = 0; i < 3; i++)
    {
        if (world_rank == 0)
        {
            // token = 0; removed
            MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

            MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);

            printf("FINAL TOKEN %d\n", token);
        }
        else
        {
            MPI_Recv(&token, 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);

            printf("%d has token %d sending to %d\n", world_rank, token, 
            (world_rank+1)%world_size); // modified
            token = token + 2;

            MPI_Send(&token, 1, MPI_INT, (world_rank+1)%world_size, 0, MPI_COMM_WORLD); // modified
        }

        endtime = MPI_Wtime();
        printf("That took %f seconds\n",endtime-starttime);
    }
    MPI_Finalize();

    return 0;
}