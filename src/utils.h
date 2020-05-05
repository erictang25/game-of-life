/**
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Sequential Version
 * 
 * To compile:
 * 
 * 
 * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BILLION 1000000000L
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

int check_generation_output( int *A, int *ref, int N );
void print_world_bits( uint8_t *world, int N );
int world_bits_correct( uint8_t *test, uint8_t *ref, int N );
