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
	./gol_seq

gol_parallel: $(src)/gol_parallel.cu
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel
