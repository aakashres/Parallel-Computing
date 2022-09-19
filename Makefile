.PHONY: average


configdir:
	mkdir -p ./bin

average: configdir
	gcc -o ./bin/average -lpthread average.c

parallel_average:
	gcc -fopenmp -o ./bin/parallel_average parallel_average_openmp.c

clean:
	rm -rf ./bin/*
