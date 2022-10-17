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

clean:
	rm -rf ./bin/*
