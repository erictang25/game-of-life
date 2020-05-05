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
                   num_alive = 0, north_byte, south_byte;
  int start = x*num_bytes; 
  int end   = start + num_bytes;   
  // printf("nb[%d] s[%d] e[%d] x:[%d] \n", num_bytes, start, end, x);                       
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
      
      // look west
      if ( (bit == 7) && (i % world_length != 0) ){
        // for the lsb
        if ( curr_world[i-1] & 0x1 ) num_alive++;
      }
      // look east
      if ( (bit == 0) && ((i + 1) % world_length != 0) ){
        // for the msb
        if ( (curr_world[i-1] >> 7) & 0x1 ) num_alive++;
      }
      // look north
      if ( (i > (world_length - 1)) ){
        north_byte = curr_world[i-world_length];
        if ( (north_byte >> (bit-1)) & 0x1 ) num_alive++;
        if ( (north_byte >> (bit))   & 0x1 ) num_alive++;
        if ( (north_byte >> (bit+1)) & 0x1 ) num_alive++;
      } 
      // look south
      if ( (i < (arr_length - world_length)) ){
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
      // printf("B[%d] b[%d] alive:%d num_alive:%d st:%x \n", i, bit, curr_bit, 
        // num_alive, next_8_states);
    }
    next_world[i] = next_8_states;
  }
  // __syncthreads();
  // for ( int i = x*blockDim.x*gridDim.x; i < num_bytes; i++ ){
  //   curr_world[i] = next_world[i];
  // }
}

int gol_bit_per_cell( uint8_t *world, int N, int P, int rounds, int test, 
                       uint8_t **ref, int trace ){
  int world_length = N/8;
  int num_elements = N*N/8;
  struct timespec t_start, t_end;
	long double average = 0, st, ed, diff;
  if ( P > 1024 )
  int blocks = 
  dim3 Block(1); // Square pattern
  dim3 Grid(P);
  uint8_t *dev_curr_world, *dev_next_world;
  cudaMalloc((void **) &dev_curr_world, num_elements*sizeof(uint8_t)); 
  cudaMalloc((void **) &dev_next_world, num_elements*sizeof(uint8_t)); 
  // cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
  for ( int i = 0; i < rounds; i++ ){
    clock_gettime(CLOCK_MONOTONIC, &t_start); /* Start Timer */
    cudaMemcpy(dev_curr_world, world, num_elements*sizeof(uint8_t), cudaMemcpyHostToDevice);
    gol_cycle<<<Grid, Block>>>( dev_curr_world, dev_next_world, N*N/8/P, 
                                world_length, num_elements );
    cudaMemcpy(world, dev_next_world, num_elements*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &t_end);   /* End timer */
    st = t_start.tv_sec + (long double)t_start.tv_nsec/BILLION;
    ed = t_end.tv_sec + (long double)t_end.tv_nsec/BILLION;
    diff = ed - st;
    average += diff/((long double)rounds);
    // print_world_bits(ref, N);
    if ( test && !world_bits_correct(world, ref[i], N)) return 1;
    if ( trace ) print_world_bits(world, N);
  }
  printf("1|%.13LF|\n", average);
  return 0;
}

int main( int argc, char** argv ){
  int test   = 0; // Run direct test cases
  int P      = 1; // number of threads
  int N      = 8; /* Matrix size */
  int ROUNDS = 5; /* Number of Rounds */
  int trace  = 0; // print trace of world  
  if ( argc > 1 ) test = atoi(argv[1]); 
  if ( argc > 2 ) {
    N = atoi(argv[2]); // Dimensions of the block
    if ( N % 8 != 0 ){
      printf( "Invalid N:[%d]; must be divisible by 8\n", N );
      N = 8;
    } 
  }
  if ( argc > 3 ) { 
    P = atoi(argv[3]); // number of threads
    if ( P > N*N/8 ){
      printf( "Invalid P:[%d]; Too many threads for number of elements %d\n", P, N*N/8 );
      return 1;
    }
    if ( N*N/8 % P != 0 ){
      printf( "Invalid P:[%d]; Number of threads should be a factor of %d\n", P, N*N/8 );
      return 1;
    }
  }
  if ( argc > 4 ) ROUNDS = atoi(argv[4]); 
  if ( argc > 5 ) trace  = atoi(argv[5]); 
  uint8_t *world, **ref;
  if (test){
    int num_correct = 0, num_tests = 0;
    // T1
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_1[0];
    N      = T_DIM;
    ROUNDS = T_ROUNDS - 1;
    ref    = (uint8_t**) malloc( sizeof(uint8_t*) * ROUNDS );
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_1[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T2
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_2[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_2[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T3
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_3[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_3[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T4
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_4[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_4[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T5
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_5[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_5[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T6
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_6[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_6[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T7
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_7[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_7[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;
    // T8
    num_tests++;
    printf("Running test %d\n", num_tests);
    world  = test_8[0];
    for ( int r = 0; r < ROUNDS; r++ )
      ref[r] = test_8[r+1];
    if (!gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace )) num_correct++;

    printf("%s %d/%d tests passed %s\n", KBLU, num_correct, num_tests, KNRM);
    free(ref);
  }
  else {
    srand48(1);
    int n_elements = N*N/8;
    world = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    for(int r = 0; r < N; r++){
      for(int c = 0; c < N/8; c++)
        world[r*N/8+c] = (uint8_t)(rand() % 256);
    } 
    ref   = NULL;
    if (trace) print_world_bits( world, N );
    gol_bit_per_cell( world, N, P, ROUNDS, test, ref, trace );
    free(world);
  }
  // for (int i =0; i < N; i++){
  //   printf("%d ", world[i]);
  // }
  return 0;
}
