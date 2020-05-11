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

inline __device__ uint8_t getCellState(int x, int y, uint key) {
  uint64_t index = x + 10 * y;
  return (key >> ((3 * 10 - 1) - index)) & 0x1;
}

/* Writes LUT
 * num_elements: number of elements in the LUT each thread is 
 *               responsible for
 */
__global__ void compute10x3LUTKernel(uint8_t* LUT, int num_elements) {
	uint64_t tableIndex = blockIdx.x * blockDim.x + threadIdx.x; 
   
	for (int i = 0; i < num_elements; i++){
		uint8_t resultState = 0;
		// For each cell.
		for (int x = 1; x < 9; x++) {
			// Count alive neighbors.
			uint8_t aliveCount = 0;
			for (int dx = -1; dx < 2; dx++) {
				for (int dy = -1; dy < 2; dy++) {
					aliveCount += getCellState(x + dx, 1 + dy, tableIndex);
				}
			}
		
			uint64_t centerState = getCellState(x, 1, tableIndex);
			aliveCount -= centerState;  // Do not count center cell in the sum.
		
			if (aliveCount == 3 || (aliveCount == 2 && centerState == 1)) {
				resultState |= 1 << (8 - x);
			}
		}
		LUT[tableIndex] = resultState;
		tableIndex++;
	}
}

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
