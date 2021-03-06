# Makefile for GoL
CC = gcc
CCFLAGS = -O3 -std=gnu99 -Wall
CPP = g++
CPPFLAGS = -O3 -Wall

CUDA = /usr/local/cuda-10.1/bin/nvcc
CUDAFLAGS = -arch=compute_52 -O3
LDFLAGS = -lrt #-lm

OBJ = utils.o

src = ../src
SEQ_SOURCES = $(src)/gol_seq.c $(OBJ)
SEQ_BIT_SOURCES = $(src)/gol_bits_seq.c $(OBJ)
LUT_SOURCES = $(src)/gol_lut_parallel.cu $(src)/gol_generate_lut.cu $(OBJ)
UTIL_SOURCES = $(src)/utils.c $(src)/utils.h

default: gol_seq gol_parallel

utils: $(UTIL_SOURCES)
	$(CPP) $(CPPFLAGS) -c -o utils.o $(LDFLAGS) $<

gol_seq: $(SEQ_SOURCES)
	$(CPP) $(CPPFLAGS) -o $@ $^ 

gol_seq_bits: $(SEQ_BIT_SOURCES)
	$(CPP) $(CPPFLAGS) -o $@ $(LDFLAGS) $^

gol_parallel: $(src)/gol_parallel.cu $(OBJ)
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

gol_lut_parallel: $(LUT_SOURCES)
	$(CUDA) $(CUDAFLAGS) -o $@ $^ $(LDFLAGS)

gol_lut3x6_parallel: $(src)/gol_lut3x6_parallel.cu $(OBJ)
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
	./gol_seq 9 

test_bits_parallel: clean gol_parallel
	./gol_parallel 0 8     25 2 
	./gol_parallel 0 16    25 2 
	./gol_parallel 0 32    25 2 
	./gol_parallel 0 64    25 2 
	./gol_parallel 0 128   25 2 
	./gol_parallel 0 256   25 2 
	./gol_parallel 0 512   25 2 
	./gol_parallel 0 1024  25 2 
	./gol_parallel 0 2048  25 2 
	./gol_parallel 0 4096  25 2 
	./gol_parallel 0 8192  25 2 
	./gol_parallel 0 16384 25 2 

test_lut_parallel: gol_lut_parallel
	./gol_lut_parallel 1 8 1 1 1
	./gol_lut_parallel 1 8 1 2 1

test_lut3x6_parallel: gol_lut3x6_parallel
	./gol_lut3x6_parallel 1 8 1 1 
	./gol_lut3x6_parallel 1 8 1 2 

eval: gol_lut3x6_parallel gol_lut_parallel
	python ../src/eval.py

clean:
	rm -rf core.* *.o *.gcda *.gcno gol_seq gol_parallel test gol_bits_seq *.out
