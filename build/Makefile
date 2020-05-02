# Makefile for GoL

CC = gcc
CCFLAGS = -std=gnu99 -Wall -O3

CUDA = /usr/local/cuda-10.1/bin/nvcc
CUDAFLAGS = -arch=compute_52
LDFLAGS = #-lm

src = ../src

default: gol_seq

gol_seq: $(src)/gol_seq.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

gol_parallel: $(src)/gol_parallel.cu
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel
