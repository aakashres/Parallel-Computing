.PHONY: average


configdir:
	mkdir -p ./bin

average: configdir
	gcc -o ./bin/average -lpthread average.c

fibonacci: configdir
	gcc -fopenmp -o ./bin/fibonacci fibonacci.c

parallel_average: configdir
	gcc -fopenmp -o ./bin/parallel_average parallel_average_openmp.c

mpi_parallel: configdir
	mpicc -o ./bin/mpi_parallel parallel_average_mpi.c

clean:
	rm -rf ./bin/*
