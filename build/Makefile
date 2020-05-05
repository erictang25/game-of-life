# Makefile for GoL

CC = gcc
CCFLAGS = -O3 -std=gnu99 -Wall
CPP = g++
CPPFLAGS = -O3 -Wall

CUDA = /usr/local/cuda-10.1/bin/nvcc
CUDAFLAGS = -arch=compute_52 -O3
LDFLAGS = -lrt #-lm

src = ../src

OBJ = utils.o
CUDAOBJ = cuda_utils.o

default: gol_seq

utils.o: $(src)/utils.c $(src)/utils.h 
	$(CPP) $(CPPFLAGS) -c -o $@ $(LDFLAGS) $<

gol_seq: $(src)/gol_seq.c $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	./gol_seq 1 1 1

gol_parallel: $(src)/gol_parallel.cu $(OBJ)
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

unit_test: $(src)/unit_test.c $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	# ./test

test_bits_para: clean gol_parallel
	./gol_parallel 1 8 1 
	# ./gol_parallel 1 8 4
	# ./gol_parallel 1 8 8

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel test
