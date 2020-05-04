/**
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Parallel Version
 * Parallel version using bit per cell implementation
 * 
 * To compile:
 * 
 * 
 * */

#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"

__global__ void gol_cycle( uint8_t *world, int world_N ){
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

}

__host__ void gol_bit_per_cell( uint8_t *world, int N, int rounds, 
                                int test, uint8_t *ref ){
  dim3 Block(1); // Square pattern
  dim3 Grid(1);
  uint8_t *dev_world;
  cudaMalloc((void **) &dev_world, N*sizeof(uint8_t)); 
  cudaMemcpy(dev_world, world, n_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
  for ( int i = 0; i < rounds; i++ ){
    gol_cycle<<<Grid, Block>>>( dev_world, world_N );
  }
}

int main( int argc, char** argv ){
  int N = 10;     /* Matrix size */
  int ROUNDS = 5; /* Number of Rounds */
  int test = 0;
  int P = 1; // number of threads
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) {
    N = atoi(argv[2]);
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%d]; must be divisible by 8", N );
    } 
  }
  if ( argc > 3 ) {
    P = atoi(argv[3]); 
  }
  if ( argc > 4 ) ROUNDS = atoi(argv[4]); 
  struct timespec start, end;
	double diff;
  int n_elements; // Number of elements in the world
  uint8_t *world;
  if (test){
    world = test_1[0];
    ref   = test_1[1];
    ROUNDS= T_ROUNDS;
  }
  else {
    int n_elements = N*N/8; // Number of elements in the world
    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    ref   = NULL;
  }
  gol_bit_per_cell( world, n_elements, ROUNDS, test, ref );

  return 0;
}
