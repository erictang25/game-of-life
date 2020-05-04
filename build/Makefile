# Makefile for GoL

CC = gcc
CCFLAGS = -O3 -std=gnu99 -Wall

CUDA = /usr/local/cuda-10.1/bin/nvcc
CUDAFLAGS = -arch=compute_52
LDFLAGS = -lrt #-lm

src = ../src

default: gol_seq

gol_seq: $(src)/gol_seq.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

gol_parallel: $(src)/gol_parallel.cu
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

unit_test: $(src)/unit_test.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	./test

test3: $(src)/gol_seq.c
	./gol_seq 3

test4: $(src)/gol_seq.c
	rm -rf gol_seq
	$(CC) $(CCFLAGS) -o gol_seq $^ $(LDFLAGS)
	./gol_seq 4

test6: $(src)/gol_seq.c
	rm -rf gol_seq
	$(CC) $(CCFLAGS) -o gol_seq $^ $(LDFLAGS)
	./gol_seq 6

test: $(src)/gol_seq.c
	./gol_seq 1
	./gol_seq 2 
	./gol_seq 3 
	./gol_seq 4 
	./gol_seq 5 
	./gol_seq 6 
	./gol_seq 7 
	./gol_seq 8 

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel test
