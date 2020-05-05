/**
 * Eric Tang (et396), Xiaoyu Yan (xy97) 
 * ECE 5720 Final Project Game of Life Parallel Version
 * Parallel version using bit per cell implementation
 * 
 * To compile:
 * 
 * 
 * */

// #include "cuda_profiler_api.h"
#include <stdint.h>
#include "utils.h"
#include "test_case_bits.h"
#define BUFFER_SIZE 10
/* 
 * Each byte is 8 cells
 * Each cell is one bit
 * curr_world : shared array for the entire grid
 * next_world : next shared array for the entire grid
 * num_bytes  : number of bytes to iterate through for this kernel; each byte is
 * 8 cells
 * world_length: length of the ????  
 */
__global__ void gol_cycle( uint8_t *curr_world, uint8_t *next_world, int num_bytes, 
                           int world_length, int arr_length ){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;
  register uint8_t curr_8_states, curr_bit = 0, next_8_states = 0,
                   num_alive = 0, threshold, north_byte, south_byte;
  int start = x*num_bytes; 
  int end   = start + num_bytes;   
  printf("nb[%d] s[%d] e[%d] x:[%d] \n", num_bytes, start, end, x);                       
  for ( int i = start; i < end; i++ ){
    curr_8_states = curr_world[i];
    next_8_states = curr_8_states;
    for (int bit = 0; bit < 8; bit ++){
      // iterate through each bit
      curr_bit = (curr_8_states >> bit) & 0x1; // extract cell state 
      num_alive = 0;
      // Check west and east first since they are in register
      // look west in same register
      if ( bit != 7 )
        if ( (curr_8_states >> (bit+1)) & 0x1 ) num_alive++;
      
      // look east in same register
      if ( bit != 0 )
        if ( (curr_8_states >> (bit-1)) & 0x1 ) num_alive++;
      
      threshold = curr_bit ? 2 : 3; 
      // look west
      if ( (num_alive < threshold) && (bit == 7) && (i % world_length != 0) ){
        // for the lsb
        if ( curr_world[i-1] & 0x1 ) num_alive++;
      }
      // look east
      if ( (num_alive < threshold) && (bit == 0) && ((i + 1) % world_length != 0) ){
        // for the msb
        if ( (curr_world[i-1] >> 7) & 0x1 ) num_alive++;
      }
      // look north
      if ( (num_alive < threshold) && (i > (world_length - 1)) ){
        north_byte = curr_world[i-world_length];
        if ( (north_byte >> (bit-1)) & 0x1 ) num_alive++;
        if ( (north_byte >> (bit))   & 0x1 ) num_alive++;
        if ( (north_byte >> (bit+1)) & 0x1 ) num_alive++;
      } 
      // look south
      if ( (num_alive < threshold) && (i < (arr_length - world_length)) ){
        south_byte = curr_world[i+world_length];
        if ( (south_byte >> (bit-1)) & 0x1 ) num_alive++;
        if ( (south_byte >> (bit))   & 0x1 ) num_alive++;
        if ( (south_byte >> (bit+1)) & 0x1 ) num_alive++;
      }
      // Determine if an alive cell will die 
      if (curr_bit && (num_alive < 2 || num_alive > 3)) 
        next_8_states ^= (uint8_t)(1 << bit);
      // Determine if a dead cell will be alive
      else if ( !curr_bit && (num_alive == 3)){
        next_8_states ^= (uint8_t)(1 << bit);
      }
      printf("B[%d] b[%d] alive:%d num_alive:%d st:%x \n", i, bit, curr_bit, 
        num_alive, next_8_states);
    }
    next_world[i] = next_8_states;
  }
  // __syncthreads();
  // for ( int i = x*blockDim.x*gridDim.x; i < num_bytes; i++ ){
  //   curr_world[i] = next_world[i];
  // }
}

int gol_bit_per_cell( uint8_t *world, int N, int P, int rounds, int test, 
                       uint8_t **ref ){
  int world_length = N/8;
  int num_elements = N*N/8;
  dim3 Block(1); // Square pattern
  dim3 Grid(P);
  uint8_t *dev_curr_world, *dev_next_world;
  cudaMalloc((void **) &dev_curr_world, num_elements*sizeof(uint8_t)); 
  cudaMalloc((void **) &dev_next_world, num_elements*sizeof(uint8_t)); 
  // cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
  for ( int i = 0; i < rounds; i++ ){
    cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
    gol_cycle<<<Grid, Block>>>( dev_curr_world, dev_next_world, N*N/8/P, 
                                world_length, num_elements );
    cudaMemcpy(world, dev_next_world, num_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // print_world_bits(ref, N);
    if ( test && !world_bits_correct(world, ref[i], N)) return 1;
  }
  return 0;
}

int main( int argc, char** argv ){
  int N = 8;     /* Matrix size */
  int ROUNDS = 5; /* Number of Rounds */
  int test = 0;
  int P = 1; // number of threads
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) { 
    P = atoi(argv[2]); // number of threads
    if ( P > N*N/8 ){
      printf( "Invalid P:[%d]; Too many threads for number of elements %d\n", P, N*N/8 );
      return 1;
    }
    if ( N*N/8 % P != 0 ){
      printf( "Invalid P:[%d]; Number of threads should be a factor of %d\n", P, N*N/8 );
      return 1;
    }
  }
  if ( argc > 3 ) {
    N = atoi(argv[3]); // Dimensions of the block
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%d]; must be divisible by 8\n", N );
      N = 8;
    } 
  }
  if ( argc > 4 ) ROUNDS = atoi(argv[4]); 
  // struct timespec start, end;
	// double diff;
  uint8_t *world, **ref;
  if (test){
    // T1
    printf("Running test 1\n");
    world  = test_1[0];
    N      = T_DIM;
    ROUNDS = T_ROUNDS - 1;
    ref    = (uint8_t**) malloc( sizeof(uint8_t*) * ROUNDS );
    for ( int r = 0; r < ROUNDS; r++ )
    ref[r] = test_1[r+1];
    gol_bit_per_cell( world, N, P, ROUNDS, test, ref );
    // T2
    // printf("Running test 2\n");
    // world  = test_2[0];
    // for ( int r = 0; r < ROUNDS; r++ )
    //   ref[r] = test_2[r+1];
    // gol_bit_per_cell( world, N, P, ROUNDS, test, ref );
    // // T3
    // printf("Running test 3\n");
    // world  = test_3[0];
    // for ( int r = 0; r < ROUNDS; r++ )
    //   ref[r] = test_3[r+1];
    // gol_bit_per_cell( world, N, P, ROUNDS, test, ref );
    // T8
    printf("Running test 3\n");
    world  = test_8[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_8[r+1];
    gol_bit_per_cell( world, N, P, ROUNDS, test, ref );

    free(ref);
  }
  else {
    int n_elements = N*N/8;
    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    ref   = NULL;
    print_world_bits(world, N);
    gol_bit_per_cell( world, N, P, ROUNDS, test, ref );
    free(world);
  }
  // for (int i =0; i < N; i++){
  //   printf("%d ", world[i]);
  // }
  return 0;
}
