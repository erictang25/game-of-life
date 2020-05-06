# Makefile for GoL

CC = gcc
CCFLAGS = -O3 -std=gnu99 -Wall
CPP = g++
CPPFLAGS = -O3 -Wall

CUDA = /usr/local/cuda-10.1/bin/nvcc
CUDAFLAGS = -arch=compute_52 -O3
LDFLAGS = -lrt #-lm

src = ../src
SEQ_SOURCES = ../src/gol_seq.c ../src/utils.c

OBJ = utils.o

default: gol_seq gol_parallel

utils.o: $(src)/utils.c $(src)/utils.h 
	$(CPP) $(CPPFLAGS) -c -o $@ $(LDFLAGS) $<

gol_seq: $(SEQ_SOURCES)
	$(CC) $(CCFLAGS) -o $@ $^ 

gol_bits_seq: $(src)/gol_bits_seq.c ../src/utils.c
	$(CC) $(CPPFLAGS) -o $@ $(LDFLAGS) $^

gol_parallel: $(src)/gol_parallel.cu $(OBJ)
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

seq_tests: $(SEQ_SOURCES)
	./gol_seq 1
	./gol_seq 2 
	./gol_seq 3 
	./gol_seq 4 
	./gol_seq 5 
	./gol_seq 6 
	./gol_seq 7 
	./gol_seq 8 

test_bits_para: clean gol_parallel
	./gol_parallel 1 8 1 

unit_test: $(src)/unit_test.c $(OBJ)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel test gol_bits_seq *.out
