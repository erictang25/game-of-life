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
	./gol_seq 1 1 1

gol_parallel: $(src)/gol_parallel.cu
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

unit_test: $(src)/unit_test.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	./test

test: gol_seq
	./gol_seq 1 1 1 

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel test
