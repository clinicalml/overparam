CC=g++
CFLAGS=-I.

all: sample.o sampling.o mtrand.o
	$(CC) -o sample -O3 sample.o sampling.o mtrand.o

sampler: sample.o sampling.o mtrand.o
	$(CC) -o sample -O3 sample.o sampling.o mtrand.o

clean:
	rm *.o
