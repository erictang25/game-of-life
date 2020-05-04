/**
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Parallel Version
 * Parallel version using bit per cell implementation
 * 
 * To compile:
 * 
 * 
 * */
 
 
#include "cuda_profiler_api.h"
#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"

// Each cell is one bit
// Each byte is 8 cells
__global__ void gol_cycle( uint8_t *world, int N, int world_length ){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  for ( int i = x; i < N; i+= offset )
  printf("%x\n",world[i]);
}

void gol_bit_per_cell( uint8_t *world, int N, int P, int rounds, int test, 
                       uint8_t *ref ){
  int world_length = N/8;
  int num_elements = N*N/8;
  dim3 Block(2); // Square pattern
  dim3 Grid(1);
  uint8_t *dev_world;
  cudaMalloc((void **) &dev_world, num_elements*sizeof(uint8_t)); 
  cudaMemcpy(dev_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
  // for ( int i = 0; i < rounds; i++ ){
  gol_cycle<<<Grid, Block>>>( dev_world, N, world_length );
  cudaMemcpy(world, dev_world, num_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  // }
}

int main( int argc, char** argv ){
  int N = 8;     /* Matrix size */
  int ROUNDS = 5; /* Number of Rounds */
  int test = 0;
  int P = 1; // number of threads
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) {
    N = atoi(argv[2]); // Dimensions of the block
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%d]; must be divisible by 8\n", N );
    } 
  }
  if ( argc > 3 ) { 
    P = atoi(argv[3]); // number of threads
    if ( P > N*N/8 ){
      printf( "Invalid P:[%d]; Too many threads for number of elements %d\n", P, N*N/8 );
    }
  }
  if ( argc > 4 ) ROUNDS = atoi(argv[4]); 
  struct timespec start, end;
	double diff;
  uint8_t *world, *ref;
  if (test){
    world  = test_1[0];
    ref    = test_1[1];
    N      = T_DIM;
    ROUNDS = T_ROUNDS;
  }
  else {
    int n_elements = N*N/8;
    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    ref   = NULL;
  }
  gol_bit_per_cell( world, N, P, ROUNDS, test, ref );

  return 0;
}
