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
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        int index, tmp_sum, sum=0, elements_per_process;
        int input[INPUT_SIZE];

        elements_per_process = INPUT_SIZE / size;
        for (i = 0; i < INPUT_SIZE; i++)
            input[i] = i + 1;
        if (size > 1)
        {
            for (i = 1; i < size - 1; i++)
            {
                index = i * elements_per_process;
                MPI_Send(&elements_per_process, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&input[index], elements_per_process, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            index = i * elements_per_process;
            int elements_left = INPUT_SIZE - index;
            MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&input[index], elements_left, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for (i = 0; i < elements_per_process; i++)
            sum += input[i];

        for (i = 1; i < size; i++)
        {
            MPI_Recv(&tmp_sum, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int sender = status.MPI_SOURCE;
            sum += tmp_sum;
        }

        // prints the final sum of array
        printf("Average of array is : %0.3f\n", (double)sum / INPUT_SIZE);
    }
    else
    {
        int n_elements_recieved;
        MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int recv_sub_arr[n_elements_recieved];
        MPI_Recv(&recv_sub_arr, n_elements_recieved, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // calculates its partial sum
        int partial_sum = 0;
        for (i = 0; i < n_elements_recieved; i++)
            partial_sum += recv_sub_arr[i];

        // sends the partial sum to the root process
        MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
