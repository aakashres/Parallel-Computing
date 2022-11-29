.PHONY: average


configdir:
	mkdir -p ./bin
	mkdir -p ./r2-logs

average: configdir
	gcc -o ./bin/average -lpthread average.c

fibonacci: configdir
	gcc -fopenmp -o ./bin/fibonacci fibonacci.c

parallel_average: configdir
	gcc -fopenmp -o ./bin/parallel_average parallel_average_openmp.c

mpi_parallel: configdir
	mpicc -o ./bin/mpi_parallel parallel_average_mpi.c

mpi_matrix: configdir
	mpicc -o ./bin/mpi_matrix matrix_multiply.c

cuda_matrix: configdir
	nvcc matrix_multiply.cu -o ./bin/cuda_matrix

hello_cuda: configdir
	nvcc hello_cuda.cu -o ./bin/hello_cuda

cuda_parallel: configdir
	nvcc cuda_parallel_average.cu -o ./bin/parallel_cuda	

	
sum: configdir
	nvcc sum.cu -o ./bin/parallel_cuda	

clean:
	rm -rf ./bin/*
