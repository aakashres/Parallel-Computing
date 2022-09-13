.PHONY: average


configdir:
	mkdir -p ./bin

average: configdir
	gcc -o ./bin/average -lpthread average.c

clean:
	rm -rf ./bin/*
