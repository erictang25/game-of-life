/* ECE 5720 Final Project Game of Life Parallel LUT Version
 * Parallel version using Lookup Table
 * 
 * To compile:
 *
 *
 * Authors: Eric Tang (et396), Xiaoyu Yan (xy97)
 * Date:    10 May 2020 
 */
#include <stdint.h>
#include <stdio.h>
#include "utils.h"

__global__ void computeLUTKernel(uint8_t* LUT, int num_elements) {

int main( int argc, char** argv ){
  uint64_t P = 1; // number of threads
  int width = 3;
  int length = 3;
  if ( argc > 1 ) P = atoi(argv[1]); 
  if ( argc > 2 ) width = atoi(argv[2]); 
  if ( argc > 3 ) length = atoi(argv[3]); 
  /* LUT parameters */
	uint64_t LUT_size = pow(2,30);
	uint64_t num_elements = LUT_size / P;
  int blocks = 1;
  while ( P > 1024 ){
    P -= 1024;
    blocks ++;
  }
  dim3 Block(blocks); // Square pattern
  dim3 Grid(P);

  uint8_t *dev_curr_world, *dev_next_world, *LUT;
  cudaMalloc((void **) &dev_curr_world, num_elements*sizeof(uint8_t)); 
}
