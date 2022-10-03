.PHONY: average


configdir:
	mkdir -p ./bin

average: configdir
	gcc -o ./bin/average -lpthread average.c

fibonacci: configdir
	gcc -fopenmp -o ./bin/fibonacci fibonacci.c

parallel_average:
	gcc -fopenmp -o ./bin/parallel_average parallel_average_openmp.c

clean:
	rm -rf ./bin/*
